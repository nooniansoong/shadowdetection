set boxwidth 0.9 absolute
set style fill   solid 1.00 border lt -1
set key inside right top vertical Right noreverse noenhanced autotitle nobox
set style histogram clustered gap 1 title textcolor lt -1
set datafile separator "\t"
set style data histograms
#set xtics border in scale 0,0 nomirror rotate by -45  autojustify
set xtics  norangelimit
set xtics   ()
set ytics 1000
set xlabel "Number of records"
set title "Training time (seconds)" 
set yrange [ 0.00000 : 36000. ] noreverse nowriteback
x = 0.0
i = 22
plot 'training.dat' using 2:xtic(1) ti col, '' u 3 ti col, '' u 4 ti col, '' u 5 ti col
set term svg size 1280,720
set output "performancesTraining.svg"
replot
set term x11
