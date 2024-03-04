serials           =
base_path         = 
measurement_dates = 
verbose           = 
plots             = 

def best_results: 
    uncoated_ave, uncoated_std  = average_results(serials, '/home/controls/CRIME/results/', date_uncoated, min_num_meas=1, bayesian=False)

    filename = f"/home/controls/CRIME/results/{serials[0]}/Grouped_Output_{serials[0]}_{date_uncoated[0]}.txt"

    df = calculate_group_stats_and_plot(filename,serials[0],date_uncoated,'/home/controls/CRIME/results/',verbose=False,confidence_level=0.95)
    df = df.sort_values(by=['Frequency'])
    df = df.loc[(df != 0).any(axis=1)]

    df_dict = {}  # Initialize an empty dictionary

    for date in date_uncoated:

        uncoated_ave, uncoated_std = average_results(serials, '/home/controls/CRIME/results/', [date], min_num_meas=1, bayesian=False)

        filename = f"/home/controls/CRIME/results/{serials[0]}/Grouped_Output_{serials[0]}_{date}.txt"

        temp_df = calculate_group_stats_and_plot(filename, serials[0], [date], '/home/controls/CRIME/results/', verbose=False, confidence_level=0.95)

        # Assuming the function returns a DataFrame, sort and clean it
        temp_df_sorted = temp_df.sort_values(by=['Frequency'])
        temp_df_cleaned = temp_df_sorted.loc[(temp_df_sorted != 0).any(axis=1)]

        # Store the cleaned DataFrame in the dictionary, keyed by the date
        df_dict[date] = temp_df_cleaned

    summarise(df_dict)
    output_df, figure  = merge_on_frequency_and_plot(*Df_dict.values(),verbose =False ,tolerance=1)

    # Displaying DataFrame and Plot side by side
    display_html = f"""
    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
        <div style='width: 30%;'> {output_df.to_html()} </div>
        <div style='width: 40%;'> <img src='data:image/png;base64,{fig_to_base64(figure)}'/> </div>
    </div>
    """


    best_data = {serials[0]: np.array(output_df)}

    summarise(best_data)
    # Display the HTML content
    display(HTML(display_html.strip()))    
    return output_df , best_data 
    
    
    
    
    
    