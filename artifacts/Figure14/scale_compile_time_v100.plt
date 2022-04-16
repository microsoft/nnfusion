clear
reset

set term pdfcairo monochrome transparent enhanced font "Times-New-Roman, 13" size 3.8, 1.5
set output "scale_compile_time_v100.pdf"

#set bmargin 3.5
set border 3
set boxwidth 0.8 absolute
#set xtics border in scale 0,0 nomirror rotate by -45 autojustify font "Times-New-Roman, 14"
set xtics border in scale 0,0 nomirror autojustify font "Times-New-Roman, 12" rotate by -45
set ytics border in scale 0,0 mirror norotate autojustify

#set xlabel "Matmul(M) and Conv2d(C) operators with different batch size" 
set ylabel "Compile time (s)" offset 1.5, 0

set key inside top left
set key at 0,2000000
#set key inside top right font "Times-New-Roman, 14"
#set nokey
set key maxrows 2
set style increment default
set style data lines
set style fill transparent solid 0.3 noborder
set yrange [0.001:1000000]
#set ylabel offset 1.2
#set xlabel offset 0,0.5
set logscale y

#Shadecolor = "#80E0A080"
#plot "motivation_op_perf.dat" u 1:2:6 t "Power" w yerr #, "" u 1:2 t "Theory" w lines
plot 'scale_compile_time_v100.dat'  u 2:xtic(1) t "TVM" w lp pt 6 ps 0.6 lt 1 lw 1 lc 4,\
                              '' u 3 t "Ansor" w lp pt 4 ps 0.6 lt 1 lw 1 lc 2,\
                              '' u 4 t "Roller-top1" w lp pt 9 ps 0.6 lt 1 lw 1 lc -1,\
                              '' u 5 t "Roller-top10" w lp pt 2 ps 0.6 lt 1 lw 1 lc 6

#plot "motivation_op_perf.dat"  u 1:2 t "Averaged Time" w lines lt 1 lw 2, "" u 1:3:4 with filledcurve lt 15 t "Standard Error"
			

