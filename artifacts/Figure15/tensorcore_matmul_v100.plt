
clear
reset

# set term pbm size 1000, 400
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 22" size 6, 2
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 22cm, 6cm
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 10cm, 6cm
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 8" size 2.8, 0.8
set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 3.8, 1.2

set border 3
#set logscale y
#set xtics nomirror rotate by -45
set xtics nomirror
set ytics nomirror


set ylabel "Kernel time (ms)"
#set xlabel "# of servers"
#set ylabel offset 1.5,-0.6

#set key at 5.5,115 Left reverse

set key maxrows 2
set key inside top left 
set key at 0, 150
set key spacing 0.7
set key width 0.1
#set key at 0,115
#set key reverse outside vertical bottom Left center
set style data histograms
#set style histogram cluster gap 0.4
set boxwidth 0.8
#set style fill pattern 5 border
set xtics scale 0.0

#set yrange [0:2]
set xtics
set xrange [0:5]

#unset xtics
set logscale y


set output "tensorcore_matmul_v100.pdf"
plot newhistogram lt 1, \
     'tensorcore_matmul_v100.dat' 	u 2:xtic(1) title "TF(cuBlas)" fs pattern 6 lt 1 lc 4, \
						''	u 3 title "TVM" fs pattern 7 ls -1 lc 2, \
						''	u 4 title "Roller-top1" fs solid 0.4 ls -1 lc 6, \
						''	u 5 title "Roller-top10" fs pattern 2 ls -1 lc -1
					
exit
