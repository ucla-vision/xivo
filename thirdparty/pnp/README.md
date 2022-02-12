# pnp
A RANSAC and BA based pnp wrapper for the Lambdatwist p3p solver. See: 
http://openaccess.thecvf.com/content_ECCV_2018/html/Mikael_Persson_Lambda_Twist_An_ECCV_2018_paper.html
Take a look at the benhmark at https://github.com/midjji/lambdatwist-p3p for comparisons to other methods. 


This is a simple but fast pnp solver which for most problems does not require you to think at all. 
By default it will work well in most cases, but have a look in the parameters file.

So I keep seeing people using very heavy robust loss functions for pnp, instead of a well considered ransac loop. 
In my experience the latter wins in every case except very high inlier noise, something which doesnt actually happen in practice for pnp problems. Avoid Opencvs Epnp in particular. 

