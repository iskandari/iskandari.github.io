#!/usr/bin/env Rscript
# Render the supervisor report from its canonical README and archived metrics.
args <- commandArgs(trailingOnly = FALSE)
script <- sub('^--file=', '', args[grepl('^--file=', args)])
setwd(dirname(dirname(normalizePath(script))))
stopifnot(requireNamespace('rmarkdown', quietly = TRUE), requireNamespace('jsonlite', quietly = TRUE))
stopifnot(requireNamespace('systemfonts', quietly = TRUE), requireNamespace('ragg', quietly = TRUE))
systemfonts::register_font('Report Open Sans',
  plain='docs/fonts/OpenSans.ttf', bold='docs/fonts/OpenSansBold.ttf',
  italic='docs/fonts/OpenSansItalic.ttf', bolditalic='docs/fonts/OpenSansBoldItalic.ttf')
metrics <- read.csv('summaries/crossed/comparison.csv')
fit <- jsonlite::fromJSON('summaries/crossed/fit.json')
history <- jsonlite::fromJSON('summaries/crossed/history.json')$valid_stop$k3_density_rmse
roles <- c('valid_report', 'test_temporal', 'test_spatial', 'test_spatiotemporal')
labels <- c('Historical reporting', 'Temporal test', 'Spatial test', 'Spatial + temporal')
colors <- c('#596675', '#007F83', '#CC7530', '#7861A8')
dir.create('docs/figures', showWarnings = FALSE)
plot_theme <- function(...) par(family='Report Open Sans', col.axis='#52606D', col.lab='#263744',
  col.main='#173643', fg='#CAD5DD', las=1, bty='n', ...)

ragg::agg_png('docs/figures/stopping-curve.png', width=2000, height=820, res=160)
plot_theme(mfrow=c(1,2), mar=c(4.5,4.8,3.8,1), oma=c(0,0,1,0))
for (zoom in c(FALSE, TRUE)) {
  use <- if (zoom) seq.int(100, length(history)) else seq_along(history)
  ylimits <- if(zoom) c(62.5,68) else c(60,150)
  yticks <- if(zoom) c(62.5,63:68) else seq(60,150,15)
  xticks <- c(use[1],1000,2000,3000,length(history))
  plot(use, history[use], type='n', xlim=range(use), ylim=ylimits,
       xaxs='i', yaxs='i', xaxt='n', yaxt='n',
       xlab=sprintf('Boosting round (%s–%s)',format(use[1],big.mark=','),format(length(history),big.mark=',')),
       ylab='k=3 density RMSE (birds/km³)', cex.main=1,
       main=if(zoom) 'B. Same curve, expanded RMSE scale' else 'A. Full RMSE scale')
  axis(1,at=xticks,labels=format(xticks,big.mark=',',trim=TRUE),cex.axis=.85)
  axis(2,at=yticks,labels=yticks,cex.axis=.9)
  abline(h=yticks,v=xticks,col='#E5EAEE',lty=3)
  if(!zoom) {
    rect(100,62.5,length(history),68,col='#EDF5F5',border='#9DBEC0',lty=2)
    text(2000,78,'Shaded band is expanded in panel B',cex=.7,col='#52606D')
  }
  lines(use, history[use], col='#007F83', lwd=2)
  abline(v=fit$best_rounds, col='#CC7530', lty=2, lwd=1.5)
  points(fit$best_rounds, history[fit$best_rounds], pch=21, bg='#CC7530', col='white', cex=1.2)
  legend('top', c('Stopping validation', paste('Selected round:', format(fit$best_rounds,big.mark=','))),
         col=c('#007F83','#CC7530'), lty=c(1,2), lwd=c(2,1.5), bty='n', text.col='#263744', cex=.8)
}
dev.off()

ragg::agg_png('docs/figures/density-r2-by-k.png', width=1800, height=980, res=160)
plot_theme(mar=c(4.5,4.8,2,1), oma=c(0,0,0,0))
plot(1:5, rep(NA_real_,5), type='n', ylim=c(.54,.95), xaxt='n',
     xlab='Withheld block size k (100 m layers)', ylab='Raw-density R²')
axis(1, at=1:5); grid(nx=NA,ny=NULL,col='#E5EAEE')
for (i in seq_along(roles)) {
  x <- metrics[metrics$role==roles[i] & metrics$scale=='density',]
  y <- x$r2[match(paste0('k',1:5),x$group)]
  lines(1:5,y,type='b',col=colors[i],lwd=2,pch=c(15,16,17,18)[i],lty=c(2,1,3,4)[i],cex=1.2)
}
legend('bottomleft',labels,col=colors,pch=c(15,16,17,18),lty=c(2,1,3,4),lwd=2,
       bty='n',cex=.88,text.col='#263744')
dev.off()

ragg::agg_png('docs/figures/k3-site-uncertainty.png',width=1800,height=800,res=160)
plot_theme(mar=c(4.5,12,2.5,2))
x <- metrics[metrics$group=='k3' & metrics$scale=='density',]
x <- x[match(roles,x$role),]
y <- 4:1
plot(x$equal_site_rmse,y,type='n',xlim=c(45,72),ylim=c(.5,4.6),yaxt='n',
     xlab='Equal-site k=3 density RMSE (birds/km³)',ylab='')
axis(2,at=y,labels=labels,tick=FALSE); abline(v=seq(45,70,5),col='#E5EAEE')
segments(x$equal_site_rmse_ci_low,y,x$equal_site_rmse_ci_high,y,col=colors,lwd=3)
points(x$equal_site_rmse,y,pch=21,bg=colors,col='white',cex=1.6)
text(x$equal_site_rmse,y+.25,labels=sprintf('%.2f',x$equal_site_rmse),col=colors,cex=.9)
dev.off()

font_header <- tempfile(fileext='.html')
font_files <- c('OpenSans.ttf','OpenSansBold.ttf','OpenSansItalic.ttf','OpenSansBoldItalic.ttf')
font_rules <- vapply(seq_along(font_files), function(i) sprintf(
  "@font-face{font-family:'Open Sans';font-style:%s;font-weight:%d;font-display:swap;src:url(data:font/ttf;base64,%s) format('truetype');}",
  if(i>2) 'italic' else 'normal', if(i%%2==0) 700 else 400,
  base64enc::base64encode(file.path('docs/fonts',font_files[i]))), character(1))
writeLines(c('<style>',font_rules,'</style>'),font_header)
rmarkdown::render('README.md', output_file='README.html',
  output_format=rmarkdown::html_document(toc=TRUE,toc_depth=3,theme=NULL,
    self_contained=TRUE,mathjax=NULL,highlight='textmate',code_folding='none',
    includes=rmarkdown::includes(in_header=c(font_header,'docs/handoff-style.html')),
    pandoc_args=c('--metadata','pagetitle=VPTS extrapolation — model development and generalization')),
  quiet=TRUE,encoding='UTF-8')
html <- paste(readLines('README.html',warn=FALSE,encoding='UTF-8'),collapse='\n')
pattern <- '(<(?:div|nav) id="TOC"[^>]*>)([\\s\\S]*?)(</(?:div|nav)>)'
stopifnot(grepl(pattern,html,perl=TRUE))
html <- sub(pattern,'\\1\n<details open><summary>Table of contents</summary>\\2</details>\n\\3',html,perl=TRUE)
html <- sub('(<body[^>]*>)', '\\1\n<a class="skip-link" href="#vpts-extrapolation-model-development-and-generalization">Skip to main content</a>\n<div class="main-container">', html, perl=TRUE)
html <- sub('</body>', '</div>\n</body>', html, fixed=TRUE)
writeLines(html,'README.html',useBytes=TRUE)
cat('Rendered self-contained R Markdown HTML with three figures and a collapsible contents menu.\n')
