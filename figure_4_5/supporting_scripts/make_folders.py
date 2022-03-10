import os

figure_folder = './figures/'

def make_figure_folder(output_folder):
    figure_subfolder = [e for e in output_folder.split('/') if len(e) > 0][-1]
    if figure_subfolder not in os.listdir(figure_folder):
        os.mkdir('%s/%s/'%(figure_folder, figure_subfolder))

    if 'linear_terms' not in os.listdir(output_folder):
        os.mkdir('%s/linear_terms'%(output_folder))

    if 'GSEA_output' not in os.listdir(output_folder):
        os.mkdir('%s/GSEA_output'%(output_folder))

    return '%s/%s/'%(
        figure_folder, 
        figure_subfolder
    )