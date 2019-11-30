## http://sphaerula.com/legacy/R/cobbDouglas.html
##  cobbDouglas.r
##  26-Feb-2006
##
##  Conrad Halling
##  conrad.halling@sphaerula.com
##
##  Estimate the parameters of the Cobb-Douglas economic model using data from
##  the years 1899-1922.
##
##  From:
##
##  Calculus: Concepts and Contexts, 2nd edition, by James Stewart.
##  Brooks/Cole, 2001. pp. 750-751.

##  Create a data.frame with the data for the years 1899–1922.
##  The three variables are production, labor, and capital, scaled
##  to 100 for the year 1899.

prod <- c(                                         100, 101,
           112, 122, 124, 122, 143, 152, 151, 126, 155, 159,
           153, 177, 184, 169, 189, 225, 227, 223, 218, 231,
           179, 240 )

lab  <- c(                                         100, 105,
           110, 117, 122, 121, 125, 134, 140, 123, 143, 147,
           148, 155, 156, 152, 156, 183, 198, 201, 196, 194,
           146, 161 )

cap  <- c(                                         100, 107,
           114, 122, 131, 138, 149, 163, 176, 185, 198, 208,
           216, 226, 236, 244, 266, 298, 335, 366, 387, 407,
           417, 431 )

cd   <- data.frame( prod = prod, lab = lab, cap = cap )

row.names( cd ) <- 1899:1922

##  The model proposed by Cobb and Douglas is:
##      prod = b * lab ^ alpha * cap ^ ( 1 - alpha ),
##  where b and alpha are the model parameters.
##
##  This model can be converted to a linear model by taking the log of both
##  sides of the equation:
##      log( prod ) = log( b ) + alpha * log( lab ) +
##          ( 1 - alpha ) * log( cap )
##
##  The formula estimates the two parameters we desire, where log( b ) is
##  estimated by the Intercept coefficient.

cd.lm <-
    lm(
        formula = I( log( prod ) - log( cap ) ) ~ I( log( lab ) - log( cap ) ),
        data    = cd )
summary( cd.lm )

##  Coefficients:
##                         Estimate Std. Error t value Pr(>|t|)
##  (Intercept)            0.007044   0.020134    0.35     0.73
##  I(log(lab) - log(cap)) 0.744606   0.042205   17.64  1.8e-14 ***

##  Plot the fitted surface for the ranges of values from the data.
##
##  Open the default device.

get( getOption( "device" ) )()

##  Expand the plot limits to include all the sample points.
##  The x and y coordinates of the plot are stored in variables x and y.

interval <- 10

x <-
    seq(
        from = floor( min( cd$lab ) / interval ) * interval,
        to   = ceiling( max( cd$lab ) / interval ) * interval,
        by   = interval )

y <-
    seq(
        from = floor( min( cd$cap ) / interval ) * interval,
        to   = ceiling( max( cd$cap ) / interval ) * interval,
        by   = interval )

##  Calculate the z coordinates of the plot.

##  f() is the function that predicts the z value from the x and y,
##  that is, prod from lab and cap.

f <-
function( x, y )
{
    exp( coef( cd.lm )[ 1 ] ) * ( x ^ coef( cd.lm )[ 2 ] ) *
        ( y ^ ( 1 - coef( cd.lm )[ 2 ] ) )
}

z <- outer( x, y, f )

##  Calculate the plotting limits for the z coordinate.

zlim <- c( floor( min( z ) ), ceiling( max( z ) ) )

##  Calculate the fitted values and add them to the data.frame.
##  The fitted values can also be obtained from cd5.lm using the
##  function fitted().

cd$prod.hat <- f( x = cd$lab, y = cd$cap )

##  Create a perspective plot.
##  theta is the angle of perspective for rotation around the z axis.
##  phi is the angle of perspective above the x-y plane.

theta <- -60
phi <- 10

cd.persp <-
    persp(
        x        = x,
        y        = y,
        z        = z,
        theta    = theta,
        phi      = phi,
        col      = "lightblue",
        ltheta   = -135,
        shade    = 0.5,
        ticktype = "detailed",
        main     = "Cobb-Douglas  Production Function 1899-1922",
        xlab     = "Labor",
        ylab     = "Capital",
        zlab     = "Production",
        zlim     = zlim,
        scale    = FALSE,
        border   = NA,
        nticks   = 4 )

##  Plot the data points, using the color "gray32" for points below the
##  surface and the color "black" for points above the surface.
##
##  Convert the 3-D coordinates of the points to the 2-D drawing coordinates
##  using trans3d().
##
##  Draw a line from the observed point to the corresponding fitted value
##  on the surface as an aid to visualizing the position of the point
##  in 3-D space.
##
##  First, plot the data points that are below the surface.
##  pch = 16 designates a filled circle symbol.

cd.below <- subset( cd, cd$prod < cd$prod.hat )

cd.below.trans3d <-
    trans3d(
        x    = cd.below$lab,
        y    = cd.below$cap,
        z    = cd.below$prod,
        pmat = cd.persp )

points( cd.below.trans3d, col = "gray32", pch = 16 )

##  Convert the coordinates of the fitted values.

cd.below.hat.trans3d <-
    trans3d(
        x    = cd.below$lab,
        y    = cd.below$cap,
        z    = cd.below$prod.hat,
        pmat = cd.persp )

##  Draw lines from the observed points to the fitted points.

segments(
    x0  = cd.below.trans3d$x,
    y0  = cd.below.trans3d$y,
    x1  = cd.below.hat.trans3d$x,
    y1  = cd.below.hat.trans3d$y,
    lty = "solid",
    col = "gray32" )

##  Second, plot the data points that are above the surface.

cd.above <- subset( cd, cd$prod >= cd$prod.hat )

cd.above.trans3d <-
    trans3d(
        x    = cd.above$lab,
        y    = cd.above$cap,
        z    = cd.above$prod,
        pmat = cd.persp )

points( cd.above.trans3d, col = "black", pch = 16 )

cd.above.hat.trans3d <-
    trans3d(
        x    = cd.above$lab,
        y    = cd.above$cap,
        z    = cd.above$prod.hat,
        pmat = cd.persp )

segments(
    x0  = cd.above.trans3d$x,
    y0  = cd.above.trans3d$y,
    x1  = cd.above.hat.trans3d$x,
    y1  = cd.above.hat.trans3d$y,
    lty = "solid",
    col = "black" )