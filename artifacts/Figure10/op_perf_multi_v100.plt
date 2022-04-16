
clear
reset

######### n,m: change this parameter to equal the number of data sets to be plotted
######### n: num of rows
######### m: num of columns
n = 5
m = 1
# t: top margin in pixels
t = 0.06
# b: key height in pixels (bottom margin)
b = 0.06
# h: height of output in pixels
h = 150.0*n + t + b

# l: left margin
l = 0.04
# g: gap margin
g = 0.08
# w: width
w = 10 * m


######### define functions to help set top/bottom margins
#top(i,n,h,t,b) = 1.0 - (t+(h-t-b)*(i-1)/n)/h - 0.05
#bot(i,n,h,t,b) = 1.0 - (t+(h-t-b)*i/n)/h + 0.05
top(i,n,h,t,b) = 1.0 - t - (1.0 - t) / n * i
bot(i,n,h,t,b) = 1.0 - t - (1.0 - t) / n * (i + 1) + b
lft(j,m,l,g) = l + (1.0-l+g) * j / m
rgt(j,m,l,g) = l + (1.0-l+g) * (j+1) / m - g


set style data histograms
set style histogram cluster gap 1
set boxwidth 0.5

######### set up some basic plot parameters
#set term pngcairo enhanced size 800,h font 'FreeMono-Bold,14'
#set output 'bigkey.png'
#set term pdfcairo enhanced font "Times-New-Roman, 9.5" size 4.8,2
set term pdfcairo enhanced font "Times-New-Roman, 8" size 5, 2.5
set output 'op_perf_multi_v100.pdf'

#set title 'Big Key Plot'
#NOYTICS = "set format y ''; unset ylabel"
NOYTICS = "unset ylabel"
YTICS = "set ylabel 'Time (ms)'"
#set logscale y
set xtics nomirror
set ytics nomirror
set xtics scale 0.0


set multiplot layout (n+1),m

set xtics font "Times-New-Roman, 7"
#set xrange [0.2:34.8]
set border 3

######### 0,0 plot
# change only plot command here
currentplot = 0
currentcolumn = 0
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
unset key
# set xtics scale 0.0
# set ytics scale 0.5

#set title "MatMul" font "Times-New-Roman, 8" offset 0,-12
@NOYTICS
set xrange [0.2:16]
set ytics auto
set logscale y
set xlabel offset 0,1
plot newhistogram lt 1, \
     'op_perf_multi_v100.dat' 	u 26:xtic(25) title "TF" fs solid 1 lt 1 lc 3, \
						''	u 27 title "TVM" fs solid 1 ls -1 lc 4, \
						''	u 28 title "Ansor" fs solid 1 ls -1 lc 2, \
						''	u 29 title "Roller(top1)" fs solid 1 ls -1 lc 6,\
						''	u 30 title "Roller(top10)" fs solid 1 ls -1 lc -1
######### 0,1 plot
currentplot = currentplot + 1
currentcolumn = 0
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
#set title "Conv2D" font "Times-New-Roman, 9.5" offset 0,-12
@NOYTICS
set xrange [0:27]
unset logscale y

#set yrange [0:4]
#set ytics 1
#set xlabel offset 0,1
plot newhistogram lt 1, \
     'op_perf_multi_v100.dat' 	u 2:xtic(1) title "TF" fs solid 1 lt 1 lc 3, \
						''	u 3 title "TVM" fs solid 1 ls -1 lc 4, \
						''	u 4 title "Ansor" fs solid 1 ls -1 lc 2, \
						''	u 5 title "Roller(top1)" fs solid 1 ls -1 lc 6,\
						''	u 6 title "Roller(top10)" fs solid 1 ls -1 lc -1

######### 1,0 plot
# change only plot command here
currentplot = currentplot + 1
currentcolumn = 0
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
unset key
set xtics scale 0.0
set ytics scale 0.5

#set title "DepthwiseConv2D" font "Times-New-Roman, 9.5" offset 0,-12
@NOYTICS
unset yrange
set ytics 1
#set ylabel offset 1.5
set xlabel offset 0,2
plot newhistogram lt 1, \
     'op_perf_multi_v100.dat' 	u 8:xtic(7) title "TF" fs solid 1 lt 1 lc 3, \
						''	u 9 title "TVM" fs solid 1 ls -1 lc 4, \
						''	u 10 title "Ansor" fs solid 1 ls -1 lc 2, \
						''	u 11 title "Roller(top1)" fs solid 1 ls -1 lc 6,\
						''	u 12 title "Roller(top10)" fs solid 1 ls -1 lc -1

######### 1,1 plot
currentplot = currentplot + 1
currentcolumn = 0
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)

@NOYTICS
set yrange [0:3]
#set xrange [0.2:17.8]
set ytics 1
#set logscale y
set xlabel offset 0,1
plot newhistogram lt 1, \
     'op_perf_multi_v100.dat' 	u 14:xtic(13) title "TF" fs solid 1 lt 1 lc 3, \
						''	u 15 title "TVM" fs solid 1 ls -1 lc 4, \
						''	u 16 title "Ansor" fs solid 1 ls -1 lc 2, \
						''	u 17 title "Roller(top1)" fs solid 1 ls -1 lc 6,\
						''	u 18 title "Roller(top10)" fs solid 1 ls -1 lc -1

######### 1,1 plot
currentplot = currentplot + 1
currentcolumn = 0
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)

@NOYTICS
#unset yrange 
#set xrange [0.2:17.8]
set ytics 1
#set logscale y
set xlabel offset 0,1
plot newhistogram lt 1, \
     'op_perf_multi_v100.dat' 	u 20:xtic(19) title "TF" fs solid 1 lt 1 lc 3, \
						''	u 21 title "TVM" fs solid 1 ls -1 lc 4, \
						''	u 22 title "Ansor" fs solid 1 ls -1 lc 2, \
						''	u 23 title "Roller(top1)" fs solid 1 ls -1 lc 6,\
						''	u 24 title "Roller(top10)" fs solid 1 ls -1 lc -1

unset logscale y
######### key plot
unset title
#set font "Times-New-Roman, 8"
set key maxrows 1
set tmargin at screen 1.0
set bmargin at screen top(0,n,h,t,b) - 0.05
set lmargin at screen 0
set rmargin at screen 1.0
set key center center
set border 0
unset tics
unset xlabel
unset ylabel
set yrange [0:1]

set key spacing 0.5
set key width 0.5
#set key above vertical maxrows 2

plot	0 with boxes title "TF(CudaLib)" fs solid 1 lt 1 lc 3, \
		0 with boxes title "TVM" fs solid 1 ls -1 lc 4, \
		0 with boxes title "Ansor" fs solid 1 ls -1 lc 2, \
		0 with boxes title "Roller-Top1" fs solid 1 ls -1 lc 6,\
		0 with boxes title "Roller-Top10" fs solid 1 ls -1 lc -1
######### key plot

unset multiplot
