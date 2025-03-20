import pandas as pd
import plotly.express as px


# plot vol features to see spikes in volume and maybe trend
def plot_timeseries(df, columns, title="Time Series Plot", height=600, width=1000, 
                    date_column='timestamp', log_scale=False, separate_plots=False):
    """
    Plot one or multiple time series from a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the time series data
    columns : str or list of str
        Column name(s) to plot
    title : str
        Plot title
    height : int
        Plot height in pixels
    width : int
        Plot width in pixels
    date_column : str
        Name of the column containing datetime information
    log_scale : bool
        Whether to use log scale for y-axis
    separate_plots : bool
        If True, create separate subplots for each series; if False, plot all on same axes
    
    Returns:
    --------
    fig : plotly figure
        The plotly figure object that can be displayed or saved
    """
    if isinstance(columns, str):
        columns = [columns]  # Convert single column to list
        
    if separate_plots:
        # Create separate subplots for each column
        fig = px.line(df, x=date_column, y=columns, facet_row="variable", 
                     title=title, height=height*len(columns)//2, width=width)
        fig.update_yaxes(matches=None)  # Allow different scales for each subplot
        
        # Update layout for better readability
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
    else:
        # Plot all columns on the same axes
        fig = px.line(df, x=date_column, y=columns, title=title, height=height, width=width)
    
    # Apply log scale if requested
    if log_scale:
        fig.update_yaxes(type="log")
    
    # Improve layout
    fig.update_layout(
        legend_title_text="Series",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified"
    )
    
    return fig

# Example usage:
# Single series
# fig = plot_timeseries(data.df_dict['BTC/USDT'], 'volume_ma_2', title='BTC Volume 2-period MA')
# fig.show()

# Multiple series
# fig = plot_timeseries(data.df_dict['BTC/USDT'], 
#                      ['volume_ma_2', 'volume_ma_12', 'volume_ma_48'], 
#                      title='BTC Volume Moving Averages')
# fig.show()

# Separate plots
# fig = plot_timeseries(data.df_dict['BTC/USDT'], 
#                      ['volume_ma_2', 'volume_ma_48'], 
#                      title='BTC Volume Moving Averages', 
#                      separate_plots=True)
# fig.show()
