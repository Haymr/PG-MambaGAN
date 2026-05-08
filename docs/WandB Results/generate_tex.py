import os

latex_code = r"""\documentclass[11pt,a4paper]{article}

\usepackage{graphicx}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{caption}

\setlength\parindent{0pt}
\setlength{\parskip}{1em}

\title{\textbf{PG-MambaGAN Training Diagnostic Report} \\ \Large Epoch 10 Status Analysis \\ \vspace{0.5em} \textbf{PG-MambaGAN Eğitim Teşhis Raporu} \\ \Large 10. Epoch Durum Analizi}
\author{Autonomous Analysis System / Otonom Analiz Sistemi}
\date{\today}

\begin{document}

\maketitle

\vspace{1em}
\noindent\fbox{\parbox{\dimexpr\textwidth-2\fboxsep-2\fboxrule\relax}{
\textbf{Analysis Objective / Analiz Amacı:} \\
\textbf{EN:} This report provides a mid-training diagnostic snapshot (at Epoch 10) of the \textbf{PG-MambaGAN} model. The primary goal is to empower an analysis agent (or human reviewer) to determine whether to continue training under the current configuration, halt early, or modify the architecture/hyperparameters. Please review the quantitative metrics (loss curves, learning rates) and qualitative outputs (generated images) to formulate a final verdict. \\
\textbf{TR:} Bu rapor, \textbf{PG-MambaGAN} modelinin eğitim ortası (10. Epoch) teşhis özetini sunmaktadır. Temel amaç, analiz ajanının (veya insan inceleyicinin) mevcut yapılandırmayla eğitime devam edip etmeme, erken durdurma veya mimari/hiperparametreleri değiştirme kararı vermesini sağlamaktır. Nihai kararı vermek için lütfen nicel metrikleri (kayıp eğrileri, öğrenme oranları) ve nitel çıktıları (üretilen görüntüler) inceleyin.
}}
\vspace{1em}

\clearpage
\section{Quantitative Training Metrics (W\&B Charts) / Nicel Eğitim Metrikleri}

"""

for i in range(1, 25):
    chart_file = f"chart_{i:02d}.png"
    latex_code += f"""\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=0.95\\textwidth]{{charts/{chart_file}}}
\\caption{{Metric Chart {i} / Metrik Grafiği {i}}}
\\end{{figure}}
"""
    if i % 2 == 0:
        latex_code += "\\clearpage\n"

latex_code += r"""
\clearpage
\section{Qualitative Visual Progression / Niteliksel Görsel Gelişim}
\textbf{EN:} To evaluate visual improvements and detect potential issues like mode collapse or checkerboard artifacts, the progression of generated images across 5 key training steps is presented below.\\
\textbf{TR:} Görsel iyileşmeleri değerlendirmek ve mod çökmesi (mode collapse) veya dama tahtası (checkerboard) artefaktları gibi olası sorunları tespit etmek için, üretilen görüntülerin 5 temel eğitim adımı boyunca gelişimi aşağıda sunulmuştur.

"""

steps = ["0", "5144", "10288", "15432", "20576"]

for sample in range(1, 4):
    latex_code += f"\\subsection{{Sample {sample} Generation Over Time / Örnek {sample} Görsel Gelişimi}}\n"
    latex_code += "\\begin{figure}[!htb]\n\\centering\n"
    
    # Step 0 and 5144
    latex_code += f"""\\minipage{{0.48\\textwidth}}
\\includegraphics[width=\\linewidth]{{images/Image_{sample}_Step_0.png}}
\\caption*{{Step 0}}
\\endminipage\\hfill
\\minipage{{0.48\\textwidth}}
\\includegraphics[width=\\linewidth]{{images/Image_{sample}_Step_5144.png}}
\\caption*{{Step 5144}}
\\endminipage
\\vspace{{1em}}

"""
    # Step 10288 and 15432
    latex_code += f"""\\minipage{{0.48\\textwidth}}
\\includegraphics[width=\\linewidth]{{images/Image_{sample}_Step_10288.png}}
\\caption*{{Step 10288}}
\\endminipage\\hfill
\\minipage{{0.48\\textwidth}}
\\includegraphics[width=\\linewidth]{{images/Image_{sample}_Step_15432.png}}
\\caption*{{Step 15432}}
\\endminipage
\\vspace{{1em}}

"""
    # Step 20576 centered
    latex_code += f"""\\minipage{{0.48\\textwidth}}
\\includegraphics[width=\\linewidth]{{images/Image_{sample}_Step_20576.png}}
\\caption*{{Step 20576}}
\\endminipage
\\caption{{Progression of Image {sample} / Örnek {sample} Gelişimi}}
\\end{{figure}}
\\clearpage
"""

latex_code += r"""
\vspace{2em}

\noindent\fbox{\parbox{\dimexpr\textwidth-2\fboxsep-2\fboxrule\relax}{
\textbf{Agent Analysis Prompt: Final Verdict Required / Analiz Ajanı İstemi: Nihai Karar Gerekli} \\
\textbf{EN Instructions:}
\begin{itemize}
    \item \textbf{Loss Stability:} Are the generator and discriminator losses oscillating, diverging, or stabilizing? Review the Quantitative Metrics sections.
    \item \textbf{Visual Quality Assessment:} Do the images show significant detail improvement across the 5 steps? Are there persistent checkerboard artifacts or signs of mode collapse?
    \item \textbf{Actionable Recommendation:} Based on your analysis, should the training continue as-is, be stopped, or require architectural/hyperparameter adjustments?
\end{itemize}
\textbf{TR Talimatlar:}
\begin{itemize}
    \item \textbf{Kayıp İstikrarı:} Üretici (generator) ve ayırt edici (discriminator) kayıpları dalgalanıyor mu, uzaklaşıyor mu (diverge) yoksa dengeleniyor mu?
    \item \textbf{Görsel Kalite Değerlendirmesi:} Görüntüler 5 adım boyunca önemli detay iyileştirmeleri gösteriyor mu? Kalıcı dama tahtası (checkerboard) bozulmaları veya mod çökmesi (mode collapse) belirtileri var mı?
    \item \textbf{Uygulanabilir Öneri:} Analizinize dayanarak, eğitim olduğu gibi devam mı etmeli, durdurulmalı mı yoksa mimari/hiperparametre ayarları mı gerektiriyor?
\end{itemize}
\textbf{Provide your comprehensive final verdict and justification to the human researcher. / Kapsamlı nihai kararınızı ve gerekçenizi araştırmacıya sunun.}
}}

\nocite{*}
\bibliographystyle{unsrt}
\bibliography{bibliography}
\end{document}
"""

with open("report.tex", "w") as f:
    f.write(latex_code)
