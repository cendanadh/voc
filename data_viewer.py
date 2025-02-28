from flask import Flask, render_template, request, url_for, redirect, session, jsonify
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
import folium

app = Flask(__name__)
app.secret_key = 'AAA'

# Load CSV
csv_path = 'voc_all.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame()  # Empty DataFrame fallback

# Remove unwanted column
if 'Unnamed: 0' in df.columns:
    del df['Unnamed: 0']

# KPI Categories
BENCHMARK_KPI = [
    'enodeb_share', 'onx_avg_initialbufferinterval_overall', 'onx_avg_stalling_time_overall',
    'onx_avg_video_throughput_overall', 'onx_codingsegmentsdurationaveragehd_overall',
    'onx_codingsegmentsdurationpercentagehd_overall', 'onx_download_speed_overall',
    'onx_ecq', 'onx_sum_stalling_occurance_overall', 'onx_udp_jitter_overall',
    'onx_udp_latency_overall', 'onx_udp_packetloss_overall', 'onx_upload_speed_overall',
    'sample_qualitymatch', 'tutela_download_throughput', 'tutela_excellent_quality',
    'tutela_game_parameter', 'tutela_good_coverage', 'tutela_good_quality', 'tutela_jitter',
    'tutela_latency', 'tutela_packetloss', 'tutela_upload_throughput', 'tutela_video_score_all',
    'tutela_video_score_netflix', 'tutela_video_score_youtube'
]

USER_EXPERIENCE_KPI = [
    'cei_score', 'stream_qoe', 'web_page_qoe', 'im_qoe', 'f_sharing_qoe', 'games_qoe',
    'ccis_data_service_complaint', 'ccis_signal_complaint', 'ccis_voice_and_sms_service_complaint'
]

NETWORK_PERFORMANCE_KPI = [
    'availability_rate', 'call_setup_success_rate', 'ccsr_voice_rate', 'nr_availability_rate',
    'nr_retainability_rate', 'nr_sn_setup_success_rate', 'nr_user_throughput_dl_mbps_relactuserdl',
    'ows_availability', 'sd_setup_sr_rate', 'volte_accessibility_sr', 'volte_retainability_rate'
]

@app.route('/data_select')
def data_select():
    if df.empty:
        return "Error: The CSV file is empty or missing!", 400  

    # Get filter values
    location_id_input = request.args.get('location_id', '').strip()
    selected_months = request.args.getlist('month')
    selected_kpis = request.args.getlist('kpi')  
    page = request.args.get('page', 1, type=int)
    per_page = 20  

    # Make a copy of df for filtering
    filtered_df = df.copy().astype(str)

    # Process location_id filter
    if location_id_input:
        location_ids = [loc.strip() for loc in location_id_input.split(',') if loc.strip()]
        filtered_df = filtered_df[filtered_df['location_id'].isin(location_ids)]

    # Convert months to strings before filtering
    if selected_months:
        selected_months = [str(m) for m in selected_months]  
        filtered_df = filtered_df[filtered_df['month'].isin(selected_months)]

    # Select only relevant columns
    default_columns = df.columns[:3].tolist()  # First 3 columns dynamically
    display_columns = default_columns + selected_kpis if selected_kpis else df.columns.tolist()
    filtered_df = filtered_df[display_columns]

    # Save the filtered dataset automatically
    filtered_csv_path = "voc_all_selected.csv"
    filtered_df.to_csv(filtered_csv_path, index=False)

    # Pagination
    total_rows = len(filtered_df)
    start = (page - 1) * per_page
    end = start + per_page
    df_subset = filtered_df.iloc[start:end]

    # Convert DataFrame to an HTML table
    table_html = df_subset.to_html(classes='table table-striped', index=False)

    # Render template with filter parameters in the URL for pagination
    return render_template(
        'table.html',
        table=table_html,
        page=page,
        total_rows=total_rows,
        per_page=per_page,
        location_id=location_id_input,
        selected_months=selected_months,
        selected_kpis=selected_kpis if selected_kpis else [],  # Ensure empty list if no KPIs selected
        benchmark_kpi=BENCHMARK_KPI,
        user_experience_kpi=USER_EXPERIENCE_KPI,
        network_performance_kpi=NETWORK_PERFORMANCE_KPI
    )



@app.route('/feature-importance')
def feature_importance():
    fi_overall_df = pd.read_csv('feature_importance_overall.csv')
    filtered_df = pd.read_csv('voc_all_selected.csv')
    fi_overall_df_filtered = fi_overall_df[fi_overall_df['KPI'].isin(list(filtered_df.iloc[:,3:].columns))]
    fi_overall_df_filtered_html = fi_overall_df_filtered.to_html(classes='table table-striped', index=False)


    weight_benchmark = pd.read_csv('weight_benchmark.csv')
    weight_benchmark_filtered = weight_benchmark[weight_benchmark['KPI'].isin(fi_overall_df_filtered[fi_overall_df_filtered['Category']=='mv_all_benchmark'].iloc[:,0])]
    weight_benchmark_filtered['group_weight'] = weight_benchmark_filtered['Overall_Average_coef']/weight_benchmark_filtered['Overall_Average_coef'].sum()
    weight_benchmark_filtered_html = weight_benchmark_filtered.to_html(classes='table table-striped', index=False)

    weight_service_experience = pd.read_csv('weight_service_experience.csv')
    weight_service_experience_filtered = weight_service_experience[weight_service_experience['KPI'].isin(fi_overall_df_filtered[fi_overall_df_filtered['Category']=='mv_all_serviceexperience'].iloc[:,0])]
    weight_service_experience_filtered['group_weight'] = weight_service_experience_filtered['Overall_Average_coef']/weight_service_experience_filtered['Overall_Average_coef'].sum()
    weight_service_experience_filtered_html = weight_service_experience_filtered.to_html(classes='table table-striped', index=False)


    weight_network_perfomance = pd.read_csv('weight_network_perfomance.csv')
    weight_network_perfomance_filtered = weight_network_perfomance[weight_network_perfomance['KPI'].isin(fi_overall_df_filtered[fi_overall_df_filtered['Category']=='mv_all_networkperformance'].iloc[:,0])]
    weight_network_perfomance_filtered['group_weight'] = weight_network_perfomance_filtered['Overall_Average_coef']/weight_network_perfomance_filtered['Overall_Average_coef'].sum()
    weight_network_perfomance_filtered_html = weight_network_perfomance_filtered.to_html(classes='table table-striped', index=False)


    #Import ke csv
    weight_benchmark_filtered_dict = dict(zip(weight_benchmark_filtered.iloc[:, 0], weight_benchmark_filtered.iloc[:, -1]))
    weight_benchmark_filtered_dict = pd.DataFrame(weight_benchmark_filtered_dict, index=[0])
    weight_benchmark_filtered_dict.to_csv('weight_benchmark_selected.csv')

    weight_service_experience_filtered_dict = dict(zip(weight_service_experience_filtered.iloc[:, 0], weight_service_experience_filtered.iloc[:, -1]))
    weight_service_experience_filtered_dict = pd.DataFrame(weight_service_experience_filtered_dict, index=[0])
    weight_service_experience_filtered_dict.to_csv('weight_service_experience_selected.csv')

    weight_network_perfomance_filtered_dict = dict(zip(weight_network_perfomance_filtered.iloc[:, 0], weight_network_perfomance_filtered.iloc[:, -1]))
    weight_network_perfomance_filtered_dict = pd.DataFrame(weight_network_perfomance_filtered_dict, index=[0])
    weight_network_perfomance_filtered_dict.to_csv('weight_network_performance_selected.csv')

    return render_template('feature_importance.html', table1=fi_overall_df_filtered_html, table2=weight_benchmark_filtered_html, table3=weight_service_experience_filtered_html, table4=weight_network_perfomance_filtered_html, message=None)


@app.route('/clustering')
def clustering():
    filtered_df = pd.read_csv('voc_all_selected.csv')
    
    
    if 'operator' in filtered_df:
        del filtered_df['operator']

    agg_dict = {}
    for col in filtered_df.columns:
        if col in ['ccis_data_service_complaint', 'ccis_signal_complaint', 'ccis_voice_and_sms_service_complaint']:
            # Use sum for these specific columns
            agg_dict[col] = 'sum'
        elif col not in ['location_id','month','location']:  # Exclude the grouping columns
            # Use mean for the other columns
            agg_dict[col] = 'mean'



    filtered_df_agg = filtered_df.groupby(['location_id', 'location']).agg(agg_dict)
    filtered_df_agg = filtered_df_agg.reset_index()
    filtered_df_agg_html = filtered_df_agg.to_html(classes='table table-striped', index=False)
    row_counter = len(filtered_df_agg)

    threshold_fix = pd.read_csv('threshold_fix.csv')
    threshold_fix = dict(zip(threshold_fix.iloc[:, 0], threshold_fix.iloc[:, -1]))

    def scoring_outer(x):
        col_list = list(filtered_df_agg.columns)
        col_list = [col for col in col_list if col not in ['location_id','month','location']]
        def scoring(y):
            if col in ['onx_avg_stalling_time_overall', 'tutela_latency', 'tutela_jitter', 'tutela_packetloss','onx_udp_packetloss_overall', 'ccis_data_service_complaint', 'ccis_signal_complaint', 'ccis_voice_and_sms_service_complaint']:
                return 100 if y <= threshold_fix[col] else threshold_fix[col]/y*100
            else:
                return 100 if y >= threshold_fix[col] else y/threshold_fix[col]*100
        for col in col_list:
            x[col] = x[col].apply(scoring)

    filtered_df_agg_score = filtered_df_agg.copy()
    scoring_outer(filtered_df_agg_score)
    filtered_df_agg_score_html = filtered_df_agg_score.to_html(classes='table table-striped', index=False)

  
    filtered_df_agg_group_score = filtered_df_agg_score.copy()
    filtered_df_agg_group_score['benchmark_overall'] = 0
    filtered_df_agg_group_score['service_experience_overall'] = 0
    filtered_df_agg_group_score['network_performance_overall'] = 0

    

    weight_benchmark_filtered_dict = pd.read_csv('weight_benchmark_selected.csv')
    weight_benchmark_filtered_dict = dict(zip(list(weight_benchmark_filtered_dict.columns), weight_benchmark_filtered_dict.iloc[0,:]))
    del weight_benchmark_filtered_dict['Unnamed: 0']
    for x in weight_benchmark_filtered_dict:
        filtered_df_agg_group_score[x] = filtered_df_agg_group_score[x] * weight_benchmark_filtered_dict[x]
        filtered_df_agg_group_score['benchmark_overall'] = filtered_df_agg_group_score['benchmark_overall'] + filtered_df_agg_group_score[x]

    weight_service_experience_filtered_dict = pd.read_csv('weight_service_experience_selected.csv')
    weight_service_experience_filtered_dict = dict(zip(list(weight_service_experience_filtered_dict.columns), weight_service_experience_filtered_dict.iloc[0,:]))
    del weight_service_experience_filtered_dict['Unnamed: 0']
    for x in weight_service_experience_filtered_dict:
        filtered_df_agg_group_score[x] = filtered_df_agg_group_score[x] * weight_service_experience_filtered_dict[x]
        filtered_df_agg_group_score['service_experience_overall'] = filtered_df_agg_group_score['service_experience_overall'] + filtered_df_agg_group_score[x]

    weight_network_perfomance_filtered_dict = pd.read_csv('weight_network_performance_selected.csv')
    weight_network_perfomance_filtered_dict = dict(zip(list(weight_network_perfomance_filtered_dict.columns), weight_network_perfomance_filtered_dict.iloc[0,:]))
    del weight_network_perfomance_filtered_dict['Unnamed: 0']
    for x in weight_network_perfomance_filtered_dict:
        filtered_df_agg_group_score[x] = filtered_df_agg_group_score[x] * weight_network_perfomance_filtered_dict[x]
        filtered_df_agg_group_score['network_performance_overall'] = filtered_df_agg_group_score['network_performance_overall'] + filtered_df_agg_group_score[x]


    filtered_df_agg_group_score = filtered_df_agg_group_score.loc[:,['location_id','location','benchmark_overall', 'service_experience_overall', 'network_performance_overall']]
    filtered_df_agg_group_score_html = filtered_df_agg_group_score.to_html(classes='table table-striped', index=False)



    #Graph 1
    plt.figure(figsize=(5, 3))
    plt.hist(filtered_df_agg_group_score['benchmark_overall'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], edgecolor='black')
    plt.title('benchmark_overall')
    plt.ylabel('Frequency')
    
    # Save the plot into a BytesIO object (in memory)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)  # Rewind the file pointer to the start
    
    # Encode the image as a base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    plt.clf()



    #Graph 2
    plt.hist(filtered_df_agg_group_score['service_experience_overall'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], edgecolor='black')
    plt.title('service_experience_overall')
    plt.ylabel('Frequency')
    
    # Save the plot into a BytesIO object (in memory)
    img2 = BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)  # Rewind the file pointer to the start
    
    # Encode the image as a base64 string
    img_base64_2 = base64.b64encode(img2.getvalue()).decode('utf-8')

    plt.clf()



    #Graph 3
    plt.hist(filtered_df_agg_group_score['network_performance_overall'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], edgecolor='black')
    plt.title('network_performance_overall')
    plt.ylabel('Frequency')
    
    # Save the plot into a BytesIO object (in memory)
    img3 = BytesIO()
    plt.savefig(img3, format='png')
    img3.seek(0)  # Rewind the file pointer to the start
    
    # Encode the image as a base64 string
    img_base64_3 = base64.b64encode(img3.getvalue()).decode('utf-8')

    plt.clf()




    #Clustering
    benchmark_mean = filtered_df_agg_group_score['benchmark_overall'].mean()
    service_experience_mean = filtered_df_agg_group_score['service_experience_overall'].mean()
    network_performance_mean = filtered_df_agg_group_score['network_performance_overall'].mean()
    filtered_df_agg_group_score['benchmark_class'] = filtered_df_agg_group_score['benchmark_overall'].apply(lambda x: 'good' if x/benchmark_mean >= 1.1 else 'bad' if x/benchmark_mean < 0.9 else 'average')
    filtered_df_agg_group_score['service_experience_class'] = filtered_df_agg_group_score['service_experience_overall'].apply(lambda x: 'good' if x/service_experience_mean >= 1.1 else 'bad' if x/service_experience_mean < 0.9 else 'average')
    filtered_df_agg_group_score['network_performance_class'] = filtered_df_agg_group_score['network_performance_overall'].apply(lambda x: 'good' if x/network_performance_mean >= 1.1 else 'bad' if x/network_performance_mean < 0.9 else 'average')
    filtered_df_agg_group_score['result'] = filtered_df_agg_group_score['benchmark_class'] + ' ' + filtered_df_agg_group_score['service_experience_class'] + ' ' + filtered_df_agg_group_score['network_performance_class']
    filtered_df_agg_group_score_cluster = filtered_df_agg_group_score.iloc[:,-4:]
    filtered_df_agg_group_score_cluster_html = filtered_df_agg_group_score_cluster.to_html(classes='table table-striped', index=False)

    #Map
    kabupaten_lat_long = pd.read_csv('kabupaten_lat_long.csv')
    kabupaten_lat_long = kabupaten_lat_long.iloc[:,[0,2,3]]
    kabupaten_lat_long_clustering = pd.merge(kabupaten_lat_long, filtered_df_agg_group_score.iloc[:,[0,1,-1]], on='location_id', how='inner')
    #kabupaten_lat_long_clustering_html = kabupaten_lat_long_clustering.to_html(classes='table table-striped', index=False)

    map = folium.Map(location=[-6.1676, 106.7673], zoom_start=11, tiles='CartoDB.Voyager')
    for lat, long, name, result in zip(kabupaten_lat_long_clustering['latitude'], kabupaten_lat_long_clustering['longitude'], kabupaten_lat_long_clustering['location'], kabupaten_lat_long_clustering['result']):
        colorz = 'blue'
        if result in ['good good good', 'good good average', 'good average good', 'average good good']:
            colorz = 'lightgreen'
        elif result in ['good average average', 'average good average', 'average average good']:
            colorz = 'green'

        elif 'bad' in result.split():
            colorz = 'red'
        else:
            colorz = 'lightred'
        
        
        folium.Marker([lat, long], popup=name, icon=folium.Icon(color=colorz), width="50%", height="200px").add_to(map)

    #map.save("indonesia_map_with_point.html")
    map_html = map._repr_html_() 







    return render_template('clustering.html', table1=filtered_df_agg_html, table2=filtered_df_agg_score_html, table3=filtered_df_agg_group_score_html, row_counter=row_counter, img_data=img_base64, img_data2=img_base64_2, img_data3=img_base64_3, table4=filtered_df_agg_group_score_cluster_html, map_html=map_html,message=None)





if __name__ == '__main__':
    app.run(debug=True)
