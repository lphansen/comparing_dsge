#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file contains the code for the Jupyter widgets. It is not required
for the model framework. The widgets are purely for decorative purposes.
"""

#######################################################
#####                Dependencies                 #####
#######################################################

from ipywidgets import widgets, Layout, Button, HBox, VBox, interactive
from IPython.core.display import display
from IPython.display import clear_output, Markdown, Latex
from collections import OrderedDict
import mfr.modelSoln as mc
from IPython.display import Javascript
import json
import itertools
import numpy as np

## Using globals for in place updates. This may not be the best option,
## but given the nature of the Jupyter notebook, it should suffice.

global defaultParams
defaultParams = mc.settings.paramsDefault;
global defaultModel
defaultModel = mc.Model(mc.paramsDefault)
global selectedMoments

#######################################################
#####    Helper functions for latex printout      #####
#######################################################

## Modified the fucntion from
## @yanlend on Github: https://gist.github.com/yanlend/068d690effa1361ebb94

def printMomentsLatex(moments, rows, cols, col_orientation='c'):
    """
    Print momentsinto a LateX table
.
    moments : dict of numpy array with moments
           each numpy array has shape  (N,)
    rows: list of row names, length (N,)
    cols: list of column names and keys to dicts, length (N,)
    """
    # Header
    print_str = '\\begin{array}{ l' + '|' + (" "+col_orientation + '|') * (len(cols) - 1) + 'c} \hline\n'
    for c in cols:
        print_str += ' & ' + '\\text{' + c + '}'
    print_str += ' \\\\ \n \hline'

    # Content
    for m in range(len(rows)):
        print_str += '\n\\text{' + rows[m] + '}'
        for c in range(len(cols)):
            print_str += ' &\n '
            print_str += '\hphantom{0}' + str( '{:.4f}'.format(moments[rows[m]][c])) + '\hphantom{0}'
        print_str = print_str.replace('!', '')
        print_str += "\n \\\\"    # Footer
        print_str += '\n \hline'
    print_str += ' \n \end{array}'

    # Write output
    return(print_str)

def printCorrsLatex(corrs, rows, var2Label, label2Sym, stateVarList, col_orientation='c'):
    """
    Print correlations into a LateX table
    corrs        : dict of correlations
    row          : list of row names, length (N,)
    var2Label    : dict that converts programmatic variable names to English labels
    label2Sym    : dict that converts programmatic English labels to Greek symbols
    stateVarList : list of state variables
    """

    ## Step 1: Fix row names

    #### Doing this because we want state variables to be the first elements
    #### that appear in the matrix
    rowNames = [x for x in rows if x not in stateVarList]
    rowNames = stateVarList + rowNames

    ## Step 2: Start generaitng latex code

    ### Header
    #### notice that num of rows = number of columns and the row name is the
    #### same as the column name.

    print_str = '\\begin{array}{ l' + '|' + (" "+col_orientation + '|') * (len(rows) - 1) + 'c} \hline\n'

    for c in rowNames:
        ## due to the symmetric nature, the row name is also the col name.
        print_str += ' & ' + '' + label2Sym[var2Label[c + '()']] + ''
    print_str += ' \\\\ \n \hline'

    # Content
    for n in rowNames:
        print_str += '\n ' + label2Sym[var2Label[n + '()']] + ''
        for m in rowNames:
            key     = ((n,m) if (n,m) in corrs.keys() else (m,n))
            corrNum = (1.0 if n == m else corrs[key] )
            print_str += ' &\n '
            print_str += '\hphantom{0}' + str( '{:.4f}'.format( corrNum )) + '\hphantom{0}'
        print_str = print_str.replace('!', '')
        print_str += "\n \\\\"    # Footer
        print_str += '\n \hline'
    print_str += ' \n \end{array}'

    # Write output
    return(print_str)

#######################################################
#####      Jupyter widgets for user inputs        #####
#######################################################

## This section creates the widgets that will be diplayed and used by the user
## to input parameter values.

style_mini = {'description_width': '5px'}
style_short = {'description_width': '100px'}
style_med = {'description_width': '200px'}
style_long = {'description_width': '200px'}

layout_mini =Layout(width='10%')
layout_50 =Layout(width='50%')
layout_med =Layout(width='70%')

widget_layout = Layout(width = '100%')
nu = widgets.BoundedFloatText( ## fraction of new borns
    value=0.1,
    min = -2,
    max = 2,
    step=0.1,
    disabled=False,
    description = 'Expert Pop\'n Fraction',
    style=style_med
)
lambda_d = widgets.BoundedFloatText( ## death rate
    value=0.02,
    min = -2,
    max = 3,
    step=0.01,
    disabled=False,
    description = 'Death rate',
    style = style_med
)
lambda_Z = widgets.BoundedFloatText( ## persistence of growth
    value=0.252,
    min = -2,
    max = 100,
    step=0.0001,
    disabled=False,
    description = 'Persistence of growth',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

lambda_V = widgets.BoundedFloatText( ## Persistence of agg. stochastic variance
    value=0.156,
    min = -2,
    max = 100,
    step=0.0001,
    disabled=False,
    description = 'Persistence of agg. stochastic variance',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

lambda_Vtilde = widgets.BoundedFloatText( ## persistence of idio vol
    value= 1.38,
    min = -2,
    max = 100,
    step=0.0001,
    disabled=False,
    description = 'Persistence of idio. stochastic variance',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

Vtilde_bar = widgets.BoundedFloatText( ## mean of idio vol
    value= 0.0,
    step=0.001,
    min = -2,
    max = 2,
    disabled=False,
    description = 'Mean of idio. stochastic variance',
    style = style_med
)

Z_bar = widgets.BoundedFloatText( ## mean of growth
    value= 0.0,
    step=0.001,
    min = -2,
    max = 2,
    disabled=False,
    description = 'Mean of growth',
    style = style_med
)


V_bar = widgets.BoundedFloatText( ## Mean of agg. stochastic variance
    value= 1.0,
    step=0.001,
    min = -2,
    max = 2,
    disabled=False,
    description = 'Mean of agg. stochastic variance',
    style = style_med
)


delta_e = widgets.BoundedFloatText( ## rate of time preferences for experts
    value= 0.05,
    step=0.001,
    min = -2,
    max = 2,
    disabled=False,
    description = 'Experts',
    style = style_med
)

delta_h = widgets.BoundedFloatText( ## rate of time preferences for households
    value= 0.05,
    step=0.001,
    min = -2,
    max = 2,
    disabled=False,
    description = 'Households',
    style = style_med
)

a_e = widgets.BoundedFloatText( ## experts' productivity
    value= 0.14,
    step=0.001,
    min = -2,
    max = 2,
    disabled=False,
    description = 'Experts',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

a_h = widgets.BoundedFloatText( ## households' productivity
    value= -1,
    step=0.001,
    min = -2,
    max = 2,
    disabled=False,
    description = 'Households',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)


rho_e = widgets.BoundedFloatText( ## Inverse of EIS for experts
    value= 1,
    step=0.01,
    min = -2,
    max = 100,
    disabled=False,
    description = 'Experts',
    style = style_med
)

rho_h = widgets.BoundedFloatText( ## Inverse of EIS for households
    value= 1,
    step=0.01,
    min = -2,
    max = 100,
    disabled=False,
    description = 'Households',
    style = style_med
)

phi = widgets.BoundedFloatText( ## Adjustment cost parameter
    value= 3,
    step=0.01,
    min = -20,
    max = 100,
    disabled=False,
    description = 'Adjustment Cost',
    style = style_med
)

gamma_e = widgets.BoundedFloatText( ## risk aversion for experts
    value= 1,
    step=0.01,
    min = -20,
    max = 20,
    disabled=False,
    description = 'Experts',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

gamma_h = widgets.BoundedFloatText( ## risk aversion for households
    value= 1,
    step=0.01,
    min = -20,
    max = 20,
    disabled=False,
    description = 'Households',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

sigma_K_norm = widgets.BoundedFloatText( ## volatility of TFP
    value= 0.04,
    step=0.001,
    min = -1,
    max = 1,
    disabled=False,
    description = 'TFP vol',
    style = style_med
)

sigma_Z_norm = widgets.BoundedFloatText( ## volatility of g
    value= 0.0,
    step=0.001,
    min = -1,
    max = 1,
    disabled=False,
    description = 'Growth vol',
    style = style_med
)

sigma_V_norm = widgets.BoundedFloatText( ## volatility of stochastic vol
    value= 0.0,
    step=0.001,
    min = -1,
    max = 1,
    disabled=False,
    description = 'Vol of agg. stochastic variance',
    style = style_med
)

sigma_Vtilde_norm = widgets.BoundedFloatText( ## volatility of idiosyncratic vol
    value= 0.0,
    step=0.001,
    min = -1,
    max = 1,
    disabled=False,
    description = 'Vol of idio. stochastic variance',
    style = style_med
)


alpha_K = widgets.BoundedFloatText( ## depreciation
    value= 0.05,
    step=0.01,
    min = -2,
    max = 2,
    disabled=False,
    description = 'Depreciation',
    style = style_med
)

dt = widgets.BoundedFloatText( ## Time step of the outer loop
    value= 0.1,
    step=0.001,
    min = 0,
    max = 1,
    disabled=False,
    description = 'Outer loop',
    style = style_med
)

dtInner = widgets.BoundedFloatText( ## Time step of the inner loop
    value= 0.1,
    step=0.001,
    min = 0,
    max = 1,
    disabled=False,
    description = 'Inner loop',
    style = style_med
)

tol = widgets.BoundedFloatText( ## Tolerance of outer loop
    value= 0.00001,
    step= 0.000001,
    min = 0,
    max = 1,
    disabled=False,
    description = 'Outer loop',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

CGscale = widgets.BoundedFloatText( ## Tolerance of outer loop
    value= 1.0,
    step= 0.1,
    min = 0.0000000000000001,
    max = 1,
    disabled=False,
    description = 'CG Error Scaler',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

innerTol = widgets.BoundedFloatText( ## Tolerance of inner loop
    value= 0.00001,
    step= 0.000001,
    min = 0,
    max = 1,
    disabled=False,
    description = 'Inner loop',
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

maxIters = widgets.IntText( ## maximum number of outer loop iterations
    value= 4000,
    step=1,
    min = 1,
    max = 4,
    disabled=False,
    description = 'Outer loop',
    style = style_med
)

maxItersInner = widgets.IntText( ## maximum number inner loop iterations
    value= 2000000,
    step=1,
    min = 1,
    max = 4,
    disabled=False,
    description = 'Inner loop',
    style = style_med
)


nDims = widgets.IntText( ## maximum number inner loop iterations
    value= 1,
    step=1,
    min = 1,
    max = 4,
    disabled=False,
    description = 'Number of dimensions',
    style = style_med
)

cov11 = widgets.BoundedFloatText( ## cov11
    value= 1.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)

cov12 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov13 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov14 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)


cov21 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov22 = widgets.BoundedFloatText( ## cov11
    value= 1.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov23 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov24 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)


cov31 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov32 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov33 = widgets.BoundedFloatText( ## cov11
    value= 1.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov34 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)


cov41 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov42 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov43 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)

cov44 = widgets.BoundedFloatText( ## cov11
    value= 1.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini

)



nShocks = widgets.IntText( ## maximum number inner loop iterations
    value= 1,
    step=1,
    min = 1,
    max = 3,
    disabled=False,
    description = 'Number of shocks',
    style = style_med
)

nWealth = widgets.IntText( ## maximum number inner loop iterations
    value= 100,
    step=1,
    min = 10,
    max = 3000,
    disabled=False,
    description = 'Number of points in wealth',
    style = {'description_width': '260px'},
    layout = Layout(width='70%')
)
nZ = widgets.IntText( ## maximum number inner loop iterations
    value= 0,
    step=1,
    min = 10,
    max = 3000,
    disabled=False,
    description = 'Number of points in growth',
    style ={'description_width': '260px'},
    layout = Layout(width='70%')
)

nV = widgets.IntText( ## maximum number inner loop iterations
    value= 0,
    step=1,
    min = 10,
    max = 3000,
    disabled=False,
    description = 'Number of points in agg. stochastic variance',
    style = {'description_width': '260px'},
    layout = Layout(width='70%')
)

nVtilde = widgets.IntText( ## maximum number inner loop iterations
    value= 0,
    step=1,
    min = 10,
    max = 3000,
    disabled=False,
    description = 'Number of points in idio. stochastic variance',
    style = {'description_width': '260px'},
    layout = Layout(width='70%')
)


chiUnderline = widgets.BoundedFloatText( ## volatility of g
    value= 0.5,
    step=0.001,
    min = 0,
    max = 1,
    disabled=False,
    description = 'Experts\' minimum equity retention',
    style = style_long
)

hhCap = widgets.Dropdown(
    options={'Yes','No'},
    value= 'No',
    description='Can households produce',
    disabled=False,
    style = {'description_width': '260px'},
    layout = Layout(width='70%')
)


def displayHHProd(hhCap):
    ## This function displays the box to input households productivity
    ## if hosueholds are allowed to hold capital.
    if hhCap == 'Yes':
        a_h.layout.display = None
        a_h.value = 0.5 * a_e.value
        display(a_h)
    else:
        a_h.layout.display = 'none'
        a_h.value = -1

hhCapOut = widgets.interactive_output(displayHHProd, {'hhCap': hhCap})



folderName = widgets.Text(
    value='defaultModel',
    placeholder='defaultModel',
    description='Folder name',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

loadSolution = widgets.Dropdown(
    options={'Yes','No'},
    value= 'No',
    description='Load solution from other models:',
    disabled=False,
    style = style_long
)

loadParameters = widgets.Dropdown(
    options={'Yes','No'},
    value= 'No',
    description='Load parameters from folder:',
    disabled=False,
    style = style_long
)

equityIssList = ['Expert\'s constant equity retention constraint', 'Minimum expert\'s equity retention constraint']
equityIss = widgets.Select(
    options= equityIssList,
    value='Minimum expert\'s equity retention constraint',
    description='',
    disabled=False, layout=Layout(height='50px', width='70%')
)

methodList    = ['Explicit Scheme', 'Implicit Scheme']
method   = widgets.Select(
    options=methodList,
    value='Implicit Scheme',
    description='Explicit vs Implicit',
    disabled=False,
    style = {'description_width': '150px'},
    layout=Layout( height='50px', width='70%')
)

exportFreq = widgets.IntText( ## maximum number inner loop iterations
    value= 10000,
    step=1,
    min = 1,
    max = 3000000000,
    disabled=False,
    description = 'Export Frequency',
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

preLoad = widgets.Text(
    value='defaultModel',
    placeholder='defaultModel',
    description='Solution folder',
    disabled=False,
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

preLoadParameters = widgets.Text(
    value='defaultModel',
    placeholder='defaultModel',
    description='Parameters folder',
    disabled=False,
    style = {'description_width': '230px'},
    layout = Layout(width='70%')
)

driftsNetWorth = widgets.Dropdown(
    options={1,0},
    value= 0,
    description='Compute drifts networth: 1 - portfolio choice; 0 - mkt clearing',
    disabled=False,
    style = style_long
)


overwrite = widgets.Dropdown(
    options = {'Yes', 'No'},
    value = 'Yes',
    description='Overwrite if folder exists:',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

pctsVsPts = widgets.ToggleButtons(
    options=['Percentiles', 'Points'],
    description='Percentiles/Points:',
    disabled=False,
    button_style='',
    style = style_med,
    tooltips=['Choosing starting points of shock elasticities based on the percentiles of the distribution', \
    'Choosing starting points directly by inputing the values of the points'],
)

xaxisStateDropdown = widgets.Dropdown(
    options=['Wealth Share', 'Growth', 'Stochastic Vol', 'Idio. Vol'],
    value = 'Wealth Share',
    disabled=False,
    description = 'X-axis:'
)


WPcts = widgets.Text(
    value='0.1, 0.5, 0.9',
    placeholder='0.1, 0.5, 0.9',
    description='Wealth',
    disabled=False,
    style = style_med,
    layout = layout_med
)

ZPcts = widgets.Text(
    value='0.1, 0.5, 0.9',
    placeholder='0.1, 0.5, 0.9',
    description='Growth',
    disabled=False,
    style = style_med,
    layout = layout_med
)

VPcts = widgets.Text(
    value='0.1, 0.5, 0.9',
    placeholder='0.1, 0.5, 0.9',
    description='Stochastc vol.',
    disabled=False,
    style = style_med,
    layout = layout_med
)

VtildePcts = widgets.Text(
    value='0.1, 0.5, 0.9',
    placeholder='0.1, 0.5, 0.9',
    description='Idio. vol.',
    disabled=False,
    style = style_med,
    layout = layout_med
)

T = widgets.IntText( ## maximum number inner loop iterations
    value= 100,
    step=1,
    min = 1,
    max = 3000,
    disabled=False,
    description = 'Total Time Steps',
    style = style_med,
    layout = layout_med
)


dtShockElas = widgets.BoundedFloatText( ## Time step of the outer loop
    value= 1.0,
    step=0.001,
    min = 0,
    max = 2,
    disabled=False,
    description = 'Time Step (Years)',
    style = style_med,
    layout = layout_med
)

perturbVar = widgets.Dropdown(
    options = defaultModel.perturbLabel2Var,
    value = 'C',
    disabled=False,
    description = 'Perturbed Variable',
    style = style_med,
    layout = layout_med
)

perturbVarComputed = widgets.Dropdown(
    options = ['Consumption'],
    value = 'Consumption',
    disabled=False,
    description = 'Perturbed Variable',
    style = style_med,
    layout = layout_med
)


momentsBox = widgets.SelectMultiple(
    options=list(defaultModel.label2Var.keys()),
    value=['Risk Price (Experts): TFP Shock'],
    rows=7,
    description='Equilibrium Quantities',
    disabled=False,
    style = style_long,
    layout = layout_med
)

updateParams = widgets.Button(
    description='Update parameters',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

runModel = widgets.Button(
    description='Run model',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

computeStatDent = widgets.Button(
    description='Compute density',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

smoothResults = widgets.Button(
    description='Smooth results',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

launchNotebook = widgets.Button(
    description='Launch notebook',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

displayPlotPanel = widgets.Button(
    description='Update Panel Charts',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

pctsBox = widgets.SelectMultiple(
    options=list(defaultModel.label2Var.keys()),
    value=['Risk Price (Experts): TFP Shock'],
    rows=7,
    description='Equilibrium Quantities',
    disabled=False,
    style = style_long,
    layout = layout_med
)


computeMomentsButton = widgets.Button(
    description='Compute Moments',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

computeCorrelationsButton = widgets.Button(
    description='Compute Correlations',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

plotVar1 = widgets.Dropdown(
    options = defaultModel.label2Var,
    value = 'piE1()',
    disabled=False,
    description = 'Left'
)


plotVar2 = widgets.Dropdown(
    options=defaultModel.label2Var,
    value='piH1()',
    disabled=False,
    description = 'Right'
)

displayShockElasButton = widgets.Button(
    description='Display Shock Elasticities Panel',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    layout = layout_50
)

displayShockElasPanelButton = widgets.Button(
    description='Display Shock Elasticities Plot',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    layout = layout_50
)


computeShockElasButton = widgets.Button(
    description='Compute Shock Elasticities',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    layout = layout_50
)



box_layout       = Layout(width='90%', justify_content='space-between')
box_layout_wide  = Layout(width='100%', justify_content='space-between')
box_layout_small = Layout(width='10%')

OLG_box = VBox([widgets.Label(value="OLG"),nu, lambda_d])
Persistence_box = VBox([widgets.Label(value="Persistence"), lambda_Z, lambda_V, lambda_Vtilde], \
layout = Layout(width='90%'))
Mean_box = VBox([widgets.Label(value="Long Run Means"), Z_bar, V_bar, Vtilde_bar], \
layout = Layout(width='90%'))
rho_box = VBox([widgets.Label(value="Rate of time preferences"), delta_e, delta_h], \
layout = Layout(width='90%'))
a_box = VBox([widgets.Label(value="Productivity"), a_e, hhCapOut, hhCap], \
layout = Layout(width='90%'))
psi_box = VBox([widgets.Label(value="Inverse of EIS"), rho_e, rho_h], \
layout = Layout(width='90%'))
gamma_box = VBox([widgets.Label(value="Risk aversion"), gamma_e, gamma_h], \
layout = Layout(width='90%'))
other_box = VBox([widgets.Label(value="Other production parameters"), phi, alpha_K], \
layout = Layout(width='90%'))
shock_box = VBox([widgets.Label(value="Shocks"), nShocks])
vol_box = VBox([widgets.Label(value="Volatility parameters"), sigma_K_norm, sigma_Z_norm, sigma_V_norm, sigma_Vtilde_norm ], \
layout = Layout(width='90%'))
dt_box = VBox([widgets.Label(value="Time steps"), dt, dtInner], \
layout = Layout(width='90%'))
tol_box = VBox([widgets.Label(value="Tolerance"), tol, innerTol, CGscale], \
layout = Layout(width='90%'))
iter_box = VBox([widgets.Label(value="Max num. of iterations"), maxIters, maxItersInner], \
layout = Layout(width='90%'))
fin_box = VBox([widgets.Label(value="Financial constraints"), chiUnderline], \
layout = Layout(width='90%'))
equityIss_box = VBox([widgets.Label(value="Expert’s Equity-Issuance Constraint"), equityIss], \
layout = Layout(width='90%'))
method_box = VBox([widgets.Label(value="Finite Difference Numerical Scheme"), method], \
layout = Layout(width='90%'))
disc_box = VBox([widgets.Label(value="Discretization"), nWealth, nZ, nV, nVtilde], \
layout = Layout(width='90%'))
out_box = VBox([widgets.Label(value="Output"), folderName, overwrite, exportFreq], \
layout = Layout(width='90%'))



def displayLoadSolution(loadSolution):
    ## This function displays the box to model solution
    ## from other models
    if loadSolution == 'Yes':
        preLoad.layout.display = None
        preLoad.value = 'defaultModel'
        display(preLoad)
    else:
        preLoad.layout.display = 'none'
        preLoad.value = 'None'

def displayLoadParameters(loadParameters):
    ## This function displays the box to load parameters
    ## from .json file
    if loadParameters == 'Yes':
        preLoadParameters.layout.display = None
        preLoadParameters.value = 'defaultModel'
        display(preLoadParameters)
    else:
        preLoadParameters.layout.display = 'none'
        preLoadParameters.value = 'None'


def displayShocks(nShocks):
    ## This function displays the box to input the local correlation matrix
    ## depending on the number of shocks (matrix size should be nShocks by nShocks)
    vboxList = []
    if nShocks > 1:
        for s in range(1,nShocks):
            boxRow = [eval('cov' + str(s + 1) + str(int(shockNum + 1))) for shockNum in range(s)]
            vboxList.append(HBox(boxRow))
        line1 = widgets.Label(value="Local Correlation Matrix")
        line2 = widgets.Label(value="(You only need to fill in the coordinates starting at the second shock. Refer to documentation.)")
        cor_box = VBox([line1, line2] + vboxList)
        display(cor_box)
    else:
        msg = widgets.Label(value="Correlation: No need to configure the matrix when there is only one shock.")
        display(msg)

def displayShockElas(stateVarList):
    vboxList = []
    for stateVar in stateVarList:
        vboxList.append(eval(stateVar + 'Pcts'))
    vboxList = vboxList + [pctsVsPts, T, dtShockElas, perturbVar]
    shockElas_box = VBox(vboxList)
    display(shockElas_box)

def displayShockElasPanel(stateVarList):
    vboxList = []
    for stateVar in stateVarList:
        vboxList.append(eval(stateVar + 'Pcts'))
    perturbVarComputed.options = [defaultModel.var2PerturbLabel[x] for x in list(defaultModel.expoElasMap.keys())]
    vboxList = vboxList + [perturbVarComputed]
    shockElasDisplay_box = VBox(vboxList)
    display(shockElasDisplay_box)


line1      = HBox([OLG_box])#, layout = box_layout)
line2      = HBox([Mean_box, Persistence_box])#, layout = box_layout)
line2_1    = HBox([vol_box, disc_box])#, layout = box_layout)
line3      = HBox([rho_box,a_box])#, layout = box_layout)
line4      = HBox([psi_box,gamma_box])#, layout = box_layout)
line8      = HBox([shock_box])#, layout = box_layout)
line6      = HBox([dt_box,tol_box])#, layout = box_layout)
line7      = HBox([iter_box, method_box])#, layout = box_layout)
line8      = HBox([shock_box])#, layout = box_layout)
equityLine = HBox([fin_box, equityIss_box])#, layout = box_layout)
paramsPanel = VBox([line1, line2, line2_1, line3, line4, other_box, line6, line7, equityLine])
run_box = VBox([widgets.Label(value="Execute Model"), updateParams, runModel, smoothResults, computeStatDent, displayPlotPanel])



#######################################################
#####                  Functions                  #####
#######################################################


def updateParamsFn(b):
    ## This is the function triggered by the updateParams button. It will
    ## modify dictionary params.
    clear_output() ## clear the output of the existing print-out
    display(run_box) ## after clearing output, re-display buttons

    ## Step 1: Check if there's a need to load parameters

    if not preLoadParameters.value == 'None':

        ## Load in parameters
        fileName = preLoadParameters.value + '/parameters.json'
        with open(fileName) as f:
            loadedParams = json.load(f)
        defaultModel.params.update(loadedParams)

        ## Reflect the changes in the boxes
        nu.value                    = defaultModel.params['nu_newborn']
        lambda_d.value              = defaultModel.params['lambda_d']
        lambda_Z.value              = defaultModel.params['lambda_Z']
        lambda_V.value              = defaultModel.params['lambda_V']
        lambda_Vtilde.value              = defaultModel.params['lambda_Vtilde']
        Vtilde_bar.value                 = defaultModel.params['Vtilde_bar']
        Z_bar.value                 = defaultModel.params['Z_bar']
        V_bar.value                 = defaultModel.params['V_bar']
        delta_e.value               = defaultModel.params['delta_e']
        delta_h.value               = defaultModel.params['delta_h']
        a_e.value                   = defaultModel.params['a_e']
        if defaultModel.params['a_h'] > 0:
            hhCap.value = 'Yes'
        a_h.value                   = defaultModel.params['a_h']
        rho_e.value                 = defaultModel.params['rho_e']
        rho_h.value                 = defaultModel.params['rho_h']
        phi.value                   = defaultModel.params['phi']
        gamma_e.value               = defaultModel.params['gamma_e']
        gamma_h.value               = defaultModel.params['gamma_h']
        sigma_K_norm.value          = defaultModel.params['sigma_K_norm']
        sigma_Z_norm.value          = defaultModel.params['sigma_Z_norm']
        sigma_V_norm.value          = defaultModel.params['sigma_V_norm']
        sigma_Vtilde_norm.value          = defaultModel.params['sigma_Vtilde_norm']
        chiUnderline.value          = defaultModel.params['chiUnderline']
        alpha_K.value               = defaultModel.params['alpha_K']
        dt.value                    = defaultModel.params['dt']
        dtInner.value               = defaultModel.params['dtInner']
        tol.value                   = defaultModel.params['tol']
        innerTol.value              = defaultModel.params['innerTol']
        maxIters.value              = defaultModel.params['maxIters']
        maxItersInner.value         = defaultModel.params['maxItersInner']
        equityIss.value             = equityIssList[int(defaultModel.params['equityIss'] - 1)]
        method.value                = methodList[int(defaultModel.params['method'] - 1)]
        #######Parameters of the grid#######
        nShocks.value                = defaultModel.params['nShocks']
        nWealth.value                = defaultModel.params['nWealth']
        nZ.value                     = defaultModel.params['nZ']
        nV.value                     = defaultModel.params['nV']
        nVtilde.value                     = defaultModel.params['nVtilde']

        #######Parameters of the input/output#######
        folderName.value                 = defaultModel.params['folderName']
        overwrite.value                  = defaultModel.params['overwrite']
        exportFreq.value                 = defaultModel.params['exportFreq']
        CGscale.value                    = defaultModel.params['CGscale']

        #######Parameters of correlation#######
        cov11.value                      = defaultModel.params['cov11']
        cov12.value                      = defaultModel.params['cov12']
        cov13.value                      = defaultModel.params['cov13']
        cov14.value                      = defaultModel.params['cov14']
        cov21.value                      = defaultModel.params['cov21']
        cov22.value                      = defaultModel.params['cov22']
        cov23.value                      = defaultModel.params['cov23']
        cov24.value                      = defaultModel.params['cov24']
        cov31.value                      = defaultModel.params['cov31']
        cov32.value                      = defaultModel.params['cov32']
        cov33.value                      = defaultModel.params['cov33']
        cov34.value                      = defaultModel.params['cov34']
        cov41.value                      = defaultModel.params['cov41']
        cov42.value                      = defaultModel.params['cov42']
        cov43.value                      = defaultModel.params['cov43']
        cov44.value                      = defaultModel.params['cov44']

    ## Step 2: Check if there's a need to load in guesses
    if not preLoad.value == 'None':
        defaultParams['preLoad']            = preLoad.value

    else:
        #defaultParams.pop('preLoad', None)
        defaultParams['preLoad'] = 'None'

    ## Removing guesses in dictionary
    defaultModel.params.pop('xiEGuess', None)
    defaultModel.params.pop('xiHGuess', None)
    defaultModel.params.pop('kappaGuess', None)
    defaultModel.params.pop('chiGuess', None)

    ## Update parameters
    defaultParams['nu_newborn']             = nu.value
    defaultParams['lambda_d']               = lambda_d.value
    defaultParams['lambda_Z']               = lambda_Z.value
    defaultParams['lambda_V']               = lambda_V.value
    defaultParams['lambda_Vtilde']               = lambda_Vtilde.value
    defaultParams['Vtilde_bar']                  = Vtilde_bar.value
    defaultParams['Z_bar']                  = Z_bar.value
    defaultParams['V_bar']                  = V_bar.value
    defaultParams['delta_e']                = delta_e.value
    defaultParams['delta_h']                = delta_h.value
    defaultParams['a_e']                    = a_e.value
    defaultParams['a_h']                    = a_h.value;  ###Any negative number means -infty
    defaultParams['rho_e']                  = rho_e.value;
    defaultParams['rho_h']                  = rho_h.value;
    defaultParams['phi']                    = phi.value;
    defaultParams['gamma_e']                = gamma_e.value;
    defaultParams['gamma_h']                = gamma_h.value;
    defaultParams['sigma_K_norm']           = sigma_K_norm.value;
    defaultParams['sigma_Z_norm']           = sigma_Z_norm.value;
    defaultParams['sigma_V_norm']           = sigma_V_norm.value;
    defaultParams['sigma_Vtilde_norm']           = sigma_Vtilde_norm.value;
    defaultParams['equityIss']              = int(equityIssList.index(equityIss.value) + 1)
    defaultParams['hhCap']                  = (0 if hhCap.value == 'No' else 1 );
    defaultParams['chiUnderline']           = chiUnderline.value;
    defaultParams['alpha_K']                = alpha_K.value;

    #######Parameters of the numerical algorithm#######
    defaultParams['method']                 = int(methodList.index(method.value) + 1)
    defaultParams['dt']                     = dt.value;
    defaultParams['dtInner']                = dtInner.value;
    defaultParams['tol']                    = tol.value;
    defaultParams['innerTol']               = innerTol.value;
    defaultParams['verbatim']               = -1
    defaultParams['maxIters']               = maxIters.value;
    defaultParams['maxItersInner']          = maxItersInner.value;

    #######Parameters of Pardiso#######

    defaultParams['iparm_2']              = 28;  ####Number of threads
    defaultParams['iparm_3']              = 0;   ####0: direct solver; 1: Enable preconditioned CG
    defaultParams['iparm_28']             = 0;   ####IEEE precision; 0: 64 bit; 1: 32 bit
    defaultParams['iparm_31']             = 0;   ####0: direct solver; 1: recursive solver (sym matrices only)

    #######Parameters of the grid#######
    defaultParams['nShocks']                = nShocks.value;
    defaultParams['numSds']                 = 5;
    defaultParams['nWealth']                = nWealth.value;
    defaultParams['logW']                   = -1;   ### If 1, solve model on log(w) grid
    defaultParams['nZ']                     = nZ.value;  ### Program will ignore this parameter if nDims < 2 or useG = -1
    defaultParams['nV']                     = nV.value;  ### Program will ignore this parameter if nDims < 2 or useG = 1
    defaultParams['nVtilde']                     = nVtilde.value;   ### Program will ignore this parameter if nDims < 4
    defaultParams['wMin']                   = 0.01;
    defaultParams['wMax']                   = 0.99;
    gridSizeList = [x for x in [defaultParams['nWealth'], defaultParams['nZ'], defaultParams['nV'],\
                             defaultParams['nVtilde']] if x > 0.0000001]
    defaultParams['nDims']                  = len(gridSizeList)

    #######Parameters of the input/output#######
    defaultParams['folderName']             = folderName.value
    defaultParams['overwrite']              = overwrite.value

    defaultParams['exportFreq']             = exportFreq.value
    defaultParams['CGscale']                = CGscale.value
    #######Parameters of correlation#######
    defaultParams['cov11']                  = cov11.value
    defaultParams['cov12']                  = cov12.value
    defaultParams['cov13']                  = cov13.value
    defaultParams['cov14']                  = cov14.value
    defaultParams['cov21']                  = cov21.value
    defaultParams['cov22']                  = cov22.value
    defaultParams['cov23']                  = cov23.value
    defaultParams['cov24']                  = cov24.value
    defaultParams['cov31']                  = cov31.value
    defaultParams['cov32']                  = cov32.value
    defaultParams['cov33']                  = cov33.value
    defaultParams['cov34']                  = cov34.value
    defaultParams['cov41']                  = cov41.value
    defaultParams['cov42']                  = cov42.value
    defaultParams['cov43']                  = cov43.value
    defaultParams['cov44']                  = cov44.value

    ###### Update model
    defaultModel.updateParameters(defaultParams)
    print('Parameters updated.')

def runModelFn(b):
    ## This is the function triggered by the runModel button. It will
    ## execute the .solve() method.
    print('Solving a ' + str(defaultModel.params['nDims']) + '-dimensional model...', \
    flush = True, sep ='')
    defaultModel.solve()
    defaultModel.printInfo() ## after solving model, print summary information.
    if defaultModel.status == 1:
        ## If model is successfully solved, update the shock elasticities panel
        display(Javascript("Jupyter.notebook.execute_cells([15,19,20,21,22])"))

def computeStatDentFn(b):
    ## This is the function triggered by the computeStatDent button. It will
    ## execute the .computeStatDent() method.

    print('Computing stationary density...')
    defaultModel.computeStatDent()
    print('Finished computing stationary density.')

def smoothResultsFn(b):

    print('Smoothing results...')
    defaultModel.smoothResults()
    if defaultModel.params['a_h'] > 0:
        print('Finished smoothing results.')

def launchNotebookFn(b):
    Javascript("Jupyter.notebook.execute_cells([3])")
    print('clicked')

def displayPlotPanelFn(b):
    print('Displaying charts...')
    display(Javascript("Jupyter.notebook.execute_cells([12])"))

def displayShockElasFn(b):
    display(Javascript("Jupyter.notebook.execute_cells([20])"))

def computeShockElasFn(b):
    clear_output()
    display(computeShockElasButton)
    print('Computing shock elasticities...')
    pctsDict = {}
    ptsList  = []
    for stateVar in defaultModel.stateVarList:

        pcts = eval(stateVar + 'Pcts.value')
        pcts = [float(x.replace(' ', '')) for x in pcts.split(',')]

        if pctsVsPts.value == 'Percentiles':
            pctsDict[stateVar] = pcts
        else:
            ptsList.append(pcts)

    if pctsVsPts.value == 'Percentiles':
        defaultModel.computeShockElas(pcts = pctsDict, T = T.value, \
                                                  dt = dtShockElas.value, perturb = perturbVar.value)
    else:
        if len(defaultModel.stateVarList) > 1:
            ptsList = np.matrix(list(itertools.product(*ptsList)))
        else:
            ptsList = np.matrix(ptsList).reshape([len(ptsList[0]), 1])
        defaultModel.computeShockElas(points = ptsList, T = T.value, \
                                              dt = dtShockElas.value, perturb = perturbVar.value)
    print('Finished computing.')
    display(Javascript("Jupyter.notebook.execute_cells([22])"))

def displayShockElasPanelFn(b):
    clear_output()
    displayShockElasPanel(defaultModel.stateVarList)
    display(displayShockElasPanelButton)
    pctsDict = {}
    for stateVar in defaultModel.stateVarList:
        pcts = eval(stateVar + 'Pcts.value')
        if pctsVsPts.value == 'Percentiles':
            pcts = [str(round((float(x.replace(' ', '')) * 100),1)) + '%' for x in pcts.split(',')]
        else:
            pcts = [str(x.replace(' ', '')) for x in pcts.split(',')]
        pctsDict[stateVar] = pcts
    defaultModel.plotElasPanel(defaultModel.perturbLabel2Var[perturbVarComputed.value], \
    **pctsDict)

def computeMomentsFn(b):
    selectedMoments = [defaultModel.label2Var[l] for l in list(momentsBox.value)]
    selectedMoments = [x.replace('()', '') for x in selectedMoments]
    ## Clear moments dictionaries
    defaultModel.apMoments = {}
    defaultModel.macroMoments = {}
    defaultModel.computeMoments(selectedMoments)
    clear_output()
    display(computeMomentsButton)
    display(Markdown('#### Macro Moments'))
    if not defaultModel.macroMoments:
        display(Markdown('No macroeconomic variables are selected.'))
    else:
        display(Latex(printMomentsLatex(defaultModel.macroMoments,
                                                list(defaultModel.macroMoments.keys()),
                                                ['Mean','Std'])))
    display(Markdown('#### Asset Pricing Moments'))
    if not defaultModel.apMoments:
        display(Markdown('No asset pricing variables are selected.'))
    else:
        display(Latex(printMomentsLatex(defaultModel.apMoments,
                                                list(defaultModel.apMoments.keys()),
                                                ['Mean','Std'])))

def computeCorrelationsFn(b):
    selectedMoments = [defaultModel.label2Var[l] for l in list(momentsBox.value)]
    selectedMoments = [x.replace('()', '') for x in selectedMoments]
    defaultModel.computeCorrs(selectedMoments)
    rowNames = [x for x in selectedMoments if x not in defaultModel.stateVarList]
    rowNames = defaultModel.stateVarList + rowNames

    clear_output()
    display(computeCorrelationsButton)
    display(Latex(printCorrsLatex(defaultModel.corrs, rowNames,
                                          defaultModel.var2Label, defaultModel.label2Sym,
                                          defaultModel.stateVarList, col_orientation='c')))


#######################################################
#####          Configure buttons                  #####
#######################################################

selectedMoments = []

updateParams.on_click(updateParamsFn)
runModel.on_click(runModelFn)
smoothResults.on_click(smoothResultsFn)
computeStatDent.on_click(computeStatDentFn)
displayPlotPanel.on_click(displayPlotPanelFn)
computeMomentsButton.on_click(computeMomentsFn)
computeCorrelationsButton.on_click(computeCorrelationsFn)
displayShockElasButton.on_click(displayShockElasFn)
computeShockElasButton.on_click(computeShockElasFn)
displayShockElasPanelButton.on_click(displayShockElasPanelFn)
