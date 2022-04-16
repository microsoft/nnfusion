clear
reset

set term pdfcairo  transparent enhanced font "Times-New-Roman, 14" size 3.8, 1.5
set output "op_compile_time_v100.pdf"

#set bmargin 3.5
set border 3
set boxwidth 0.8 absolute
#set xtics border in scale 0,0 nomirror rotate by -45 autojustify font "Times-New-Roman, 14"
set xtics border in scale 0,0 nomirror autojustify 
set ytics border in scale 0,0 mirror norotate autojustify

set xlabel "Operator ID (sorted by compilation time)" 
set ylabel "Compile time (s)"

set key inside top left
set key maxrows 2
set key at -10, 4000000
#set key inside top right font "Times-New-Roman, 14"
#set nokey
set style increment default
set style data lines
set style fill transparent solid 0.3 noborder
set yrange [0.001:1000000]
set ylabel offset 2
set xlabel offset 0,0.5
set logscale y

#Shadecolor = "#80E0A080"
#plot "motivation_op_perf.dat" u 1:2:6 t "Power" w yerr #, "" u 1:2 t "Theory" w lines
plot 'op_compile_time_v100.dat'  u 1:2 t "TVM" w lp pt 0 ps 1 lt 1 lw 3 lc 4,\
                              '' u 1:3 t "Ansor" w lp pt 0 lt 2 lw 3 lc 2,\
                              '' u 1:4 t "Roller-top1" w lp pt 0 lt 3 lw 3 lc -1,\
                              '' u 1:5 t "Roller-top10" w lp pt 0 lt 4 lw 3 lc 6

#plot "motivation_op_perf.dat"  u 1:2 t "Averaged Time" w lines lt 1 lw 2, "" u 1:3:4 with filledcurve lt 15 t "Standard Error"
			

