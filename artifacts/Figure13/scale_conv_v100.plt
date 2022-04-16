
clear
reset

# set term pbm size 1000, 400
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 22" size 6, 2
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 22cm, 6cm
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 10cm, 6cm
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 8" size 2.8, 0.8
set term pdfcairo monochrome transparent enhanced font "Times-New-Roman, 14" size 3.8, 1.5

set border 3
#set logscale y
#set xtics nomirror rotate by -45
set xtics nomirror
set ytics nomirror


set ylabel "Kernel time (ms)"
set xlabel "Batch size"
set ylabel offset 1.5,0

set key at 6.5,5000

set key maxrows 3
#set key inside top left 
set key spacing 0.7
set key width 0.1
#set key at 0,115
#set key reverse outside vertical bottom Left center
set style data histograms
#set style histogram cluster gap 0.1
set boxwidth 0.8
#set style fill pattern 5 border
set xtics scale 0.0

set yrange [0.1:5000]
set xtics
set xrange [0:8]

#unset xtics
set logscale y


set output "scale_conv_v100.pdf"
plot newhistogram lt 1, \
     'scale_conv_v100.dat' 	u 2:xtic(1) title "TF(CudaLib)" fs pattern 6 lt 1 lc 4, \
						''	u 3 title "TVM" fs pattern 7 ls -1 lc 2, \
						''	u 4 title "Ansor" fs solid 0.4 ls -1 lc 6, \
						''	u 5 title "Roller-top1" fs pattern 2 ls -1 lc -1 ,\
						''	u 6 title "Roller-top10" fs pattern 3 ls -1 lc -1			

exit
