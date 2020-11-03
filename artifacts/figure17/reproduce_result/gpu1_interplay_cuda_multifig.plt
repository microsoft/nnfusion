#!/usr/bin/env gnuplot
clear
reset

######### n,m: change this parameter to equal the number of data sets to be plotted
######### n: num of rows
######### m: num of columns
n = 1
m = 2
# t: top margin in pixels
t = 0.23
# b: key height in pixels (bottom margin)
b = 0.25
# h: height of output in pixels
h = 150.0*n + t + b

# l: left margin
l = 0.12
# g: gap margin
g = 0.05
# w: width
w = 100 * m


######### define functions to help set top/bottom margins
#top(i,n,h,t,b) = 1.0 - (t+(h-t-b)*(i-1)/n)/h - 0.05
#bot(i,n,h,t,b) = 1.0 - (t+(h-t-b)*i/n)/h + 0.05
top(i,n,h,t,b) = 1.0 - t - (1.0 - t) / n * i
bot(i,n,h,t,b) = 1.0 - t - (1.0 - t) / n * (i + 1) + b
lft(j,m,l,g) = l + (1.0-l+g) * j / m
rgt(j,m,l,g) = l + (1.0-l+g) * (j+1) / m - g


set style data histograms
#set style histogram cluster gap 0.1
set boxwidth 0.7

######### set up some basic plot parameters
#set term pngcairo enhanced size 800,h font 'FreeMono-Bold,14'
#set output 'bigkey.png'
set term pdfcairo monochrome enhanced font "Times-New-Roman, 8" size 2,0.9
set output 'figure17_reproduce.pdf'

#set title 'Big Key Plot'
#NOYTICS = "set format y ''; unset ylabel"
NOYTICS = "unset ylabel"
YTICS = "set ylabel 'Time (ms)'"

set xtics nomirror
set ytics nomirror
set xtics scale 0.0

set multiplot layout (n+1),m

set xtics
set xrange [0.5:2.5]
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
set xtics scale 0.0

set title "ResNeXt" font "Times-New-Roman, 9" offset 0,-10.4
@YTICS
set ytics 10
set ylabel offset 1
set xlabel offset 0,2
set yrange [0:35]
plot newhistogram lt 1, \
     'gpu1_batch_cuda_resnext2.dat' 	u 2:xtic(1) title "RammerBase-fast" fs pattern 0 lt 1 , \
						''	u 3 title "Rammer-fast" fs pattern 2 lt 1 , \
						''	u 4 title "RammerBase-select" fs solid 0.4 ls -1 , \
						''	u 5 title "Rammer-select" fs solid 1 ls -1

######### 2,0 plot
#currentplot = currentplot + 1
currentcolumn = currentcolumn + 1
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
set title "LSTM" font "Times-New-Roman, 9" offset 0,-10.4
@NOYTICS
set ytics 20
set xlabel offset 0,1
set yrange [0:80]
plot newhistogram lt 1, \
     'gpu1_batch_cuda_lstm2.dat' 	u 2:xtic(1) title "RammerBase-fast" fs pattern 0 lt 1, \
						''	u 3 title "Rammer-fast" fs pattern 2 lt 1, \
						''	u 4 title "RammerBase-select" fs solid 0.4 ls -1, \
						''	u 5 title "Rammer-select" fs solid 1 ls -1

######### key plot
unset title
set tmargin at screen 1.0
set bmargin at screen top(0,n,h,t,b)
set lmargin at screen 0
set rmargin at screen 1.0
set key center center
set border 0
unset tics
unset xlabel
unset ylabel
set yrange [0:2]

set key spacing 0.8
set key width -1
set key maxrows 2

plot	0 with boxes title "RammerBase-fast" fs pattern 0 lt 1, \
		0 with boxes title "Rammer-fast" fs pattern 2 lt 1, \
		0 with boxes title "RammerBase-select" fs solid 0.4 ls -1, \
		0 with boxes title "Rammer-select" fs solid 1 ls -1

unset multiplot
