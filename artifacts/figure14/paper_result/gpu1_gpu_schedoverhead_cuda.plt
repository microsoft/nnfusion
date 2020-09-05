clear
reset

#set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 12cm, 5cm
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 6,3
set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 3.9,1.6

set output "figure14_paper.pdf"
#set term pdfcairo size 20, 5
#set tmargin 1.5
set bmargin 3
set border 3
set boxwidth 0.8 absolute
#set xtics border in scale 0,0 nomirror rotate by -45 autojustify font "Times-New-Roman, 10"
set xtics border in scale 0,0 nomirror autojustify font "Times-New-Roman, 10"
set ytics border in scale 0.5,0 nomirror norotate autojustify

set ylabel "Time (ms)" font "Times-New-Roman, 14"

#set key reverse inside top right
set key inside top left font "Times-New-Roman, 14"
set style data histograms
#set style histogram rowstacked title rotate by -45 textcolor lt -1 offset character 0, -0.5
set style histogram rowstacked title textcolor lt -1 offset character 0, -0.5
set style fill pattern 5 border
set style line 1 lt -1 lw 1.5
#set yrange [0:80]

set ylabel offset 1.5
#set xlabel offset 0,20

plot newhistogram "ResNeXt", \
		'gpu1_gpu_schedoverhead_cuda.dat' using 2 t "Kernel time" ls 1 fill solid 0.2, '' using 3:xticlabels(1) t "Overhead" ls 1 fillstyle pattern 0, '' using 0:($2+$3):(sprintf("%2.0f\%",$4)) with labels font "Times-New-Roman,10" offset 0,0.5 notitle, \
	 newhistogram "NASNet", \
		'' using 5 notitle ls 1 fill solid 0.2, '' using 6:xticlabels(1) notitle ls 1 fillstyle pattern 0, '' using 0:($5+$6+$20):(sprintf("%2.0f\%",$7)) with labels font "Times-New-Roman,10" offset 10,0.5 notitle, \
	 newhistogram "AlexNet", \
		'' using 8 notitle ls 1 fill solid 0.2, '' using 9:xticlabels(1) notitle ls 1 fillstyle pattern 0, '' using 0:($8+$9):(sprintf("%2.0f\%",$10)) with labels font "Times-New-Roman,10" offset 20,0.5 notitle, \
	 newhistogram "DeepSpeech2", \
		'' using 11 notitle ls 1 fill solid 0.2, '' using 12:xticlabels(1) notitle ls 1 fillstyle pattern 0, '' using 0:($11+$12+$20):(sprintf("%2.0f\%",$13)) with labels font "Times-New-Roman,10" offset 30,0.5 notitle, \
	 newhistogram "LSTM", \
		'' using 14 notitle ls 1 fill solid 0.2, '' using 15:xticlabels(1) notitle ls 1 fillstyle pattern 0, '' using 0:($14+$15):(sprintf("%2.0f\%",$16)) with labels font "Times-New-Roman,10" offset 40,0.5 notitle, \
	 newhistogram "Seq2Seq", \
		'' using 17 notitle ls 1 fill solid 0.2, '' using 18:xticlabels(1) notitle ls 1 fillstyle pattern 0, '' using 0:($17+$18+$20):(sprintf("%2.0f\%",$19)) with labels font "Times-New-Roman,10" offset 50,0.5 notitle
			
exit
