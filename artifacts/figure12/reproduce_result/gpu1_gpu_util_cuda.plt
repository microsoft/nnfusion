clear
reset

# set term pbm size 1000, 400
set size 1.03, 1.1
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 22" size 6, 2
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 22cm, 6cm
#set term pdfcairo monochrome enhanced font "Times-New-Roman, 12" size 10cm, 6cm
set term pdfcairo monochrome enhanced font "Times-New-Roman, 8" size 2.8, 0.8

set border 3
#set logscale y
#set xtics nomirror rotate by -45
set xtics nomirror
set ytics nomirror


set ylabel "GPU Utilization (%)"
#set xlabel "# of servers"
set ylabel offset 1.5

#set key at 5.7,5 Left reverse

set key inside top right
set key spacing 0.7
#set key reverse outside vertical bottom Left center
set style data histograms
#set style histogram cluster gap 0.1
set boxwidth 0.8
#set style fill pattern 5 border
#set tics scale 0.0
set xrange [-0.5:5.5]
set yrange [0:110]
#set xtics
#unset xtics

set output "figure12_reproduce.pdf"
plot newhistogram lt 1, \
     'gpu1_gpu_util_cuda.dat' 	u 2:xtic(1) title "TF" fs pattern 0 lt 1, \
						''	u 3 title "RammerBase" fs solid 0.4 ls -1, \
						''	u 4 title "Rammer" fs solid 1 ls -1
						

exit
