set boxwidth 0.9 absolute
set style fill   solid 1.00 border lt -1
set key inside right top vertical Right noreverse noenhanced autotitle nobox
set style histogram clustered gap 1 title textcolor lt -1
set datafile separator "\t"
set style data histograms
set xtics border in scale 0,0 nomirror rotate by -45  autojustify
set xtics  norangelimit
set xtics   ()
set ytics 100
set ytics add ("30" 30)
set xlabel "Resolution"
set title "Prediction time (minutes)" 
set yrange [ 0.00000 : 2100. ] noreverse nowriteback
x = 0.0
i = 22
plot 'prediction.dat' using 2:xtic(1) ti col, '' u 3 ti col, '' u 4 ti col
set term svg size 1280,720
set output "performances.svg"
replot
set term x11
