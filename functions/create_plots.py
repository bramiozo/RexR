## DataViz using Bokeh
# https://bokeh.pydata.org/en/latest/docs/gallery/elements.html



def _genomic_expression(genomic_data, plot_types = [], parameters = {})
	# plot_types: 
	# clustering_heatmap, 
	# variance  alphan/num of genomes vs misclassification
	# sample id's, classification (color), proba (y-axis), likelihood (x-axis)
	# spider plot
	# correlation map


	# clustering heat map


	# variance  alphan/num of genomes vs misclassification


	# sample id's, classification (color), proba (y-axis), likelihood (x-axis)


	# spider plot, max: top-10


	# correlation map, max: top-10
	# R: ggpairs(wbcd[,c(2:11,1)], aes(color=diagnosis, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
	# labs(title="Cancer Mean")+
	# theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))
	# ggcorr(wbcd[,c(2:11)], name = "corr", label = TRUE)+
  	# theme(legend.position="none")+labs(title="Cancer Mean")+theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))
	# 

	# PCA variance plot: variance/number of components (bar + line)

	# PCA contribution plot matrix, bar charts


	# sources: 
	# https://www.kaggle.com/mirichoi0218/classification-breast-cancer-or-not-with-15-ml
	# 

	return True



def _graph_visualiser(graph_data= {}, labels = [], clustering_type = None)
	# spring-force, self-organising graph, MCL, AP

	return plot_object





