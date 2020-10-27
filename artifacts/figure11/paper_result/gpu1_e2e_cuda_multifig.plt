#!/usr/bin/env gnuplot
######### n,m: change this parameter to equal the number of data sets to be plotted
######### n: num of rows
######### m: num of columns
n = 1
m = 6
# t: top margin in pixels
t = 0.23
# b: key height in pixels (bottom margin)
b = 0.2
# h: height of output in pixels
h = 150.0*n + t + b

# l: left margin
l = 0.12
# g: gap margin
g = 0.05
# w: width
w = 100 * m


######### define functions to help set top/bottom margins
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
set term pdfcairo monochrome enhanced font "Times-New-Roman, 8" size 3.84,0.6
set output 'figure11_paper.pdf'

#set title 'Big Key Plot'
#NOYTICS = "set format y ''; unset ylabel"
NOYTICS = "unset ylabel"
YTICS = "set ylabel 'Time (ms)'"

set xtics nomirror
set ytics nomirror
set xtics scale 0.0

set multiplot layout (n+1),m

set xtics
set xrange [0.5:1.5]
set border 3

######### 0,0 plot #############################################
# change only plot command here
currentplot = 0
currentcolumn = 0
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
unset key
set xtics scale 0.0

@YTICS
set ytics 20
set ylabel offset 0.5
plot newhistogram lt 1, \
'gpu1_e2e_cuda_resnext.dat' u 2:xtic(1) title "TF" fs pattern 0 lt 1, \
						''	u 3 title "TF-XLA" fs pattern 6 ls -1, \
						''	u 4 title "TF-TRT" fs pattern 7 ls -1, \
						''	u 5 title "TVM" fs pattern 2 ls -1, \
						''	u 6 title "RammerBase" fs solid 0.4 ls -1, \
						''	u 7 title "Rammer" fs solid 1 ls -1

######### 2,0 plot #############################################
# change only plot command here
#currentplot = currentplot + 1
currentcolumn = currentcolumn + 1
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
#set xlabel 'X Axis'
set yrange [0:40]

@NOYTICS
set ytics 10
set ylabel offset 0.5
plot newhistogram lt 1, \
     'gpu1_e2e_cuda_nasnet.dat' 	u 2:xtic(1) title "TF" fs pattern 0 lt 1, \
						''	u 3 title "TF-XLA" fs pattern 6 ls -1, \
						''	u 4 title "TF-TRT" fs pattern 7 ls -1, \
						''	u 5 title "TVM" fs pattern 2 ls -1, \
						''	u 6 title "RammerBase" fs solid 0.4 ls -1, \
						''	u 7 title "Rammer" fs solid 1 ls -1

######### 2,1 plot #############################################
currentcolumn = currentcolumn + 1
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
unset title
#set xlabel 'X Axis'
set xtics
set yrange [0:2.5]

@NOYTICS
set ytics 0.5
plot newhistogram lt 1, \
     'gpu1_e2e_cuda_alexnet.dat' 	u 2:xtic(1) title "TF" fs pattern 0 lt 1, \
						''	u 3 title "TF-XLA" fs pattern 6 ls -1, \
						''	u 4 title "TF-TRT" fs pattern 7 ls -1, \
						''	u 5 title "TVM" fs pattern 2 ls -1, \
						''	u 6 title "RammerBase" fs solid 0.4 ls -1, \
						''	u 7 title "Rammer" fs solid 1 ls -1


######### 0,1 plot #############################################
currentcolumn = currentcolumn + 1
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
unset title
@NOYTICS
set ytics 10
set yrange [0:50]
plot newhistogram lt 1, \
     'gpu1_e2e_cuda_deepspeech2.dat' 	u 2:xtic(1) title "TF" fs pattern 0 lt 1, \
						''	u 3 title "TF-XLA" fs pattern 6 ls -1, \
						''	u 4 title "TF-TRT" fs pattern 7 ls -1, \
						''	u 5 title "TVM" fs pattern 2 ls -1, \
						''	u 6 title "RammerBase" fs solid 0.4 ls -1, \
						''	u 7 title "Rammer" fs solid 1 ls -1

######### 1,0 plot #############################################
# change only plot command here
#currentplot = currentplot + 1
currentcolumn = currentcolumn + 1
set tmargin at screen top(currentplot,n,h,t,b)
set bmargin at screen bot(currentplot,n,h,t,b)
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)

@NOYTICS
set ytics 40
set ylabel offset 1.5
set yrange [0:170]
plot newhistogram lt 1, \
     'gpu1_e2e_cuda_lstm.dat' 	u 2:xtic(1) title "TF" fs pattern 0 lt 1, \
						''	u 3 title "TF-XLA" fs pattern 6 ls -1, \
						''	u 4 title "TF-TRT" fs pattern 7 ls -1, \
						''	u 5 title "TVM" fs pattern 2 ls -1, \
						''	u 6 title "RammerBase" fs solid 0.4 ls -1, \
						''	u 7 title "Rammer" fs solid 1 ls -1

######### 1,1 plot
currentcolumn = currentcolumn + 1
set lmargin at screen lft(currentcolumn,m,l,g)
set rmargin at screen rgt(currentcolumn,m,l,g)
unset title
@NOYTICS
set ytics 20
set yrange [0:80]
plot newhistogram lt 1, \
     'gpu1_e2e_cuda_seq2seq.dat' 	u 2:xtic(1) title "TF" fs pattern 0 lt 1, \
						''	u 3 title "TF-XLA" fs pattern 6 ls -1, \
						''	u 4 title "TF-TRT" fs pattern 7 ls -1, \
						''	u 5 title "TVM" fs pattern 2 ls -1, \
						''	u 6 title "RammerBase" fs solid 0.4 ls -1, \
						''	u 7 title "Rammer" fs solid 1 ls -1

######### key plot #############################################
set key maxrows 1
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


plot	0 with boxes title "TF" fs pattern 0 lt 1, \
		0 with boxes title "TF-XLA" fs pattern 6 ls -1, \
		0 with boxes title "TF-TRT" fs pattern 7 ls -1, \
		0 with boxes title "TVM" fs pattern 2 ls -1, \
		0 with boxes title "RammerBase" fs solid 0.4 ls -1, \
		0 with boxes title "Rammer" fs solid 1 ls -1

unset multiplot
