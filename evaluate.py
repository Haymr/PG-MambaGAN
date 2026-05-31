"""
PG-MambaGAN — Full Evaluation Pipeline

Runs the complete Q1-grade evaluation:
    1. 2D per-slice metrics (PSNR, SSIM, RMSE, MAE)
    2. 3D volumetric assembly + NIfTI export
    3. 3D metrics (3D-SSIM, Flickering Index)
    4. Hallucination risk analysis (pyradiomics)
    5. Clinical task validation (EPI, CNR)

Usage:
    python evaluate.py \\
        --checkpoint experiments/checkpoints/best.pth \\
        --data-path /data/mayo/processed \\
        --manifest /data/mayo/processed/patient_manifest.json \\
        --output-dir experiments/evaluation
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from data.patient_manifest import PatientManifest
from data.dataset import LDCTDataset
from models.generators.vss_unet import VSSUNet
from models.generators.unet_baseline import UNetBaseline
from evaluation.metrics import compute_2d_metrics, compute_volumetric_metrics, get_body_mask
from evaluation.volumetric import VolumeAssembler
from evaluation.hallucination import HallucinationAnalyzer
from evaluation.clinical_task import ClinicalTaskValidator


def load_generator(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load generator from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    
    gen_type = config.get("model", {}).get("generator", "vss_unet")
    gc = config.get("generator", {})
    
    if gen_type == "vss_unet":
        model = VSSUNet(
            in_channels=config.get("model", {}).get("channels", 1),
            out_channels=config.get("model", {}).get("channels", 1),
            embed_dim=gc.get("embed_dim", 96),
            depths=gc.get("depths", [2, 2, 4, 2]),
            patch_size=gc.get("patch_size", 4),
            use_checkpoint=False,  # Not needed for eval
        )
    else:
        model = UNetBaseline()
    
    # Load EMA weights if available, else generator weights
    if "g_ema" in ckpt:
        model.load_state_dict(ckpt["g_ema"])
        print("  ✅ Loaded EMA generator weights")
    else:
        model.load_state_dict(ckpt["generator"])
        print("  ✅ Loaded generator weights")
    
    model.to(device).eval()
    return model


def evaluate_2d(
    generator: torch.nn.Module,
    dataset: LDCTDataset,
    device: str,
) -> Dict:
    """Run 2D per-slice evaluation on the test set."""
    print("\n" + "=" * 60)
    print("  2D Per-Slice Evaluation")
    print("=" * 60)
    
    all_metrics = {"psnr": [], "ssim": [], "rmse": [], "mae": []}
    
    loader = dataset.get_dataloader(batch_size=1, num_workers=2, drop_last=False)
    
    with torch.no_grad():
        for batch in loader:
            ldct = batch["ldct"].to(device)
            ndct = batch["ndct"].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                     enabled=(device == "cuda")):
                predicted = generator(ldct)
            
            # Compute metrics on CPU numpy
            pred_np = predicted.float().squeeze().cpu().numpy()
            ndct_np = ndct.squeeze().cpu().numpy()
            
            # Body mask to prevent background air inflation
            ndct_hu = (ndct_np + 1.0) / 2.0 * 2000.0 - 1000.0
            body_mask = get_body_mask(ndct_hu)
            
            metrics = compute_2d_metrics(pred_np, ndct_np, body_mask=body_mask)
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    # Aggregate
    results = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        results[f"{k}_mean"] = float(arr.mean())
        results[f"{k}_std"] = float(arr.std())
    
    print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"  RMSE: {results['rmse_mean']:.6f} ± {results['rmse_std']:.6f}")
    print(f"  MAE:  {results['mae_mean']:.6f} ± {results['mae_std']:.6f}")
    
    return results


def evaluate_3d(
    generator: torch.nn.Module,
    dataset: LDCTDataset,
    manifest: PatientManifest,
    output_dir: Path,
    metadata_dir: str,
    device: str,
) -> Dict:
    """Run 3D volumetric evaluation on test patients."""
    print("\n" + "=" * 60)
    print("  3D Volumetric Evaluation")
    print("=" * 60)
    
    assembler = VolumeAssembler(
        generator=generator,
        metadata_dir=metadata_dir,
        output_dir=str(output_dir / "volumes"),
    )
    
    test_patients = manifest.get_patient_ids("test")
    
    all_results = {}
    agg_metrics = {
        "psnr_3d_mean": [], "ssim_3d_mean": [], "flickering_index": []
    }
    
    for patient_id in test_patients:
        print(f"\n  Patient: {patient_id}")
        
        vol_result = assembler.assemble_patient(
            patient_id, dataset, device=device
        )
        
        # 3D metrics
        metrics = compute_volumetric_metrics(
            vol_result["pred_volume"],
            vol_result["ndct_volume"],
        )
        
        all_results[patient_id] = metrics
        
        print(f"    3D-PSNR: {metrics['psnr_3d_mean']:.2f} dB")
        print(f"    3D-SSIM: {metrics['ssim_3d_mean']:.4f}")
        print(f"    Flicker: {metrics['flickering_index']:.6f}")
        
        for k in agg_metrics:
            if k in metrics:
                agg_metrics[k].append(metrics[k])
    
    # Aggregate across patients
    summary = {}
    for k, vals in agg_metrics.items():
        if vals:
            arr = np.array(vals)
            summary[f"{k}_avg"] = float(arr.mean())
            summary[f"{k}_std"] = float(arr.std())
    
    print(f"\n  --- Across {len(test_patients)} patients ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    
    return {"per_patient": all_results, "summary": summary}


def evaluate_hallucination(
    generator: torch.nn.Module,
    dataset: LDCTDataset,
    manifest: PatientManifest,
    output_dir: Path,
    device: str,
) -> Dict:
    """Run hallucination risk analysis on test patients."""
    print("\n" + "=" * 60)
    print("  Hallucination Risk Analysis")
    print("=" * 60)
    
    analyzer = HallucinationAnalyzer()
    
    test_patients = manifest.get_patient_ids("test")
    all_reports = {}
    
    for patient_id in test_patients[:5]:  # Analyze up to 5 patients
        data_root = dataset.manifest.metadata.get("data_root", ".")
        ldct_vol, ndct_vol = dataset.get_patient_volume(patient_id, data_root)
        
        # Infer
        pred_slices = []
        with torch.no_grad():
            for i in range(ldct_vol.shape[0]):
                x = torch.from_numpy(ldct_vol[i]).unsqueeze(0).unsqueeze(0).to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                         enabled=(device == "cuda")):
                    pred = generator(x)
                pred_slices.append(pred.float().squeeze().cpu().numpy())
        
        pred_vol = np.stack(pred_slices)
        
        report = analyzer.analyze_volume(pred_vol, ndct_vol)
        all_reports[patient_id] = report
        
        print(f"  {patient_id}: {report['verdict']} "
              f"(preservation: {report['preservation_rate']:.0%})")
    
    # Save
    analyzer.save_report(all_reports, str(output_dir / "hallucination_report.json"))
    
    return all_reports


def evaluate_clinical(
    generator: torch.nn.Module,
    dataset: LDCTDataset,
    manifest: PatientManifest,
    output_dir: Path,
    device: str,
) -> Dict:
    """Run clinical task validation."""
    print("\n" + "=" * 60)
    print("  Clinical Task Validation")
    print("=" * 60)
    
    validator = ClinicalTaskValidator()
    
    test_patients = manifest.get_patient_ids("test")
    all_reports = {}
    
    for patient_id in test_patients[:5]:
        data_root = dataset.manifest.metadata.get("data_root", ".")
        ldct_vol, ndct_vol = dataset.get_patient_volume(patient_id, data_root)
        
        pred_slices = []
        with torch.no_grad():
            for i in range(ldct_vol.shape[0]):
                x = torch.from_numpy(ldct_vol[i]).unsqueeze(0).unsqueeze(0).to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                         enabled=(device == "cuda")):
                    pred = generator(x)
                pred_slices.append(pred.float().squeeze().cpu().numpy())
        
        pred_vol = np.stack(pred_slices)
        
        report = validator.validate_volume(pred_vol, ndct_vol, ldct_vol)
        all_reports[patient_id] = report
        
        print(f"  {patient_id}: {report['clinical_verdict']}")
        print(f"    EPI: {report.get('epi_mean', 0):.3f} | "
              f"CNR improvement: {report.get('cnr_improvement_pct_mean', 0):.1f}%")
    
    validator.save_report(all_reports, str(output_dir / "clinical_report.json"))
    
    return all_reports


def main():
    parser = argparse.ArgumentParser(description="PG-MambaGAN Evaluation")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--metadata-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/evaluation")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--skip-hallucination", action="store_true")
    parser.add_argument("--skip-clinical", action="store_true")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    generator = load_generator(args.checkpoint, device)
    
    # Load data
    manifest_path = args.manifest or str(Path(args.data_path) / "patient_manifest.json")
    manifest = PatientManifest.load(manifest_path)
    metadata_dir = args.metadata_dir or str(Path(args.data_path) / "metadata")
    
    test_dataset = LDCTDataset(
        manifest=manifest, split="test",
        data_root=args.data_path, augment=False,
    )
    
    # Run evaluations
    all_results = {}
    
    # 1. 2D metrics
    all_results["2d_metrics"] = evaluate_2d(generator, test_dataset, device)
    
    # 2. 3D volumetric
    if not args.skip_3d:
        all_results["3d_metrics"] = evaluate_3d(
            generator, test_dataset, manifest, output_dir, metadata_dir, device
        )
    
    # 3. Hallucination
    if not args.skip_hallucination:
        all_results["hallucination"] = evaluate_hallucination(
            generator, test_dataset, manifest, output_dir, device
        )
    
    # 4. Clinical
    if not args.skip_clinical:
        all_results["clinical"] = evaluate_clinical(
            generator, test_dataset, manifest, output_dir, device
        )
    
    # Save combined results
    with open(output_dir / "full_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"  ✅ Full evaluation complete! Results: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
