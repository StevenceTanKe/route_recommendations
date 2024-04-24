import streamlit as st
import pandas as pd
import numpy as np
import pants
import json


def setup():
    st.set_page_config(
        page_title='澳门旅游路线推荐系统',
        page_icon='data/map/icon/macau.ico',
        layout='wide'
    )

    st.title('澳门旅游路线推荐系统')
    st.write('根据您的 *旅游类型* 和 *兴趣* 推荐最适合的旅游路线。')

    st.sidebar.title("旅游路线设置")
    tour_type = st.sidebar.radio('请选择旅游类型：', ('一日游', '多日游'))

    if 'hours_spots' not in st.session_state:
        st.session_state['hours_spots'] = {}
        st.session_state['hours_spots']['day_tour'] = 1
        st.session_state['hours_spots']['multi_day'] = 1

    if tour_type == '一日游':
        st.session_state['hours_spots']['day_tour'] = st.sidebar.number_input(
            '请输入一天期望游览的小时数',
            min_value=1,
            max_value=24,
            help='小时数范围：1~24',
            value=st.session_state['hours_spots'].get('day_tour', 1),
            key='day_tour')
    elif tour_type == '多日游':
        st.session_state['hours_spots']['multi_day'] = st.sidebar.number_input(
            '请输入期望游览的景点数量',
            min_value=1,
            max_value=49,
            help='景点数量范围：1~49',
            value=st.session_state['hours_spots'].get('multi_day', 1),
            key='multi_day')

    selected_tags = st.sidebar.multiselect('请选择你感兴趣的标签：',
                                           options=['Square', 'Park', 'Street', 'Museum', 'Performance',
                                                    'Western-Church', 'Chinese', 'Landmark'])

    return tour_type, st.session_state['hours_spots']['day_tour'], st.session_state['hours_spots'][
        'multi_day'], selected_tags


def validate_weights_expectation(tour_type, day_tour_hours, multi_day_spots, selected_tags):
    if 'weights' not in st.session_state:
        st.session_state['weights'] = {}

    total_weight = 0
    if selected_tags:
        st.sidebar.write(" ")
        st.sidebar.write(" ")
        st.sidebar.write("**请输入每个标签的权重，权重总和应为 1**")

        for tag in selected_tags:
            st.session_state['weights'][tag] = st.sidebar.number_input(
                f"权重（{tag}）",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state['weights'].get(tag, 0.0),
                step=0.01,
                key=tag)
            total_weight += st.session_state['weights'][tag]

        if total_weight == 1.0:
            st.sidebar.success("权重验证成功！")

            if st.sidebar.button('推荐路线'):
                display_recommendations(tour_type, day_tour_hours, multi_day_spots, selected_tags,
                                        st.session_state['weights'])
        else:
            st.sidebar.error(f"权重和为 *{total_weight:.2f}*，请调整以满足总和为 *1* 的要求。")
            st.sidebar.button("推荐路线", disabled=True)

    return st.session_state['weights']


def display_recommendations(tour_type, day_tour_hours, multi_day_spots, selected_tags, weights):
    st.success(f"根据您选择的旅游类型 **{tour_type}** 和以下标签及权重为您推荐路线...")
    data = {'标签': [], '权重': []}
    for tag in selected_tags:
        data['标签'].append(tag)
        data['权重'].append(weights[tag])

    # 将字典转换为DataFrame
    df = pd.DataFrame(data)
    df.set_index('标签', inplace=True)  # 将标签列设置为索引
    st.table(df)

    if tour_type == '一日游':
        one_day_tour(day_tour_hours, selected_tags)
    elif tour_type == '多日游':
        multi_day_tour(multi_day_spots, selected_tags)


def one_day_tour(day_tour_hours, selected_tags):
    df_attraction = pd.read_excel('data/processed/attractions_information.xlsx')

    # 根据 selected_tags 和 weights 得到排序
    speed = 5

    label_weights = {}
    for tag in selected_tags:
        label_weights[tag] = st.session_state['weights'][tag]

    # 执行加权得分计算和排序
    df_attraction['Weighted_Score'] = 0
    for label, weight in label_weights.items():
        df_attraction['Weighted_Score'] += df_attraction.get(label, 0) * weight
    df_recommendation = df_attraction.sort_values(by='Weighted_Score', ascending=False).head(day_tour_hours + 1)

    print(df_recommendation[['Attraction', 'Weighted_Score']].to_string(index=False))
    print(f"\n Stay duration: {df_recommendation['tour_duration'].sum()} h")

    # 蚁群算法得到最短路线
    # 加载经纬度数据
    df_coords = df_attraction.loc[df_recommendation.index]

    # 创建距离矩阵
    distance_matrix = np.zeros((len(df_coords), len(df_coords)))
    for i, (idx1, row1) in enumerate(df_coords.iterrows()):
        for j, (idx2, row2) in enumerate(df_coords.iterrows()):
            distance_matrix[i, j] = haversine(row1['lat'], row1['lng'], row2['lat'], row2['lng'])

    # 创建距离矩阵的计算函数
    def distance(i, j):
        return distance_matrix[i][j]

    world = pants.World(range(len(distance_matrix)), distance)
    solver = pants.Solver(
        alpha=1.0,  # 信息素重要性
        beta=4.0,  # 启发式信息的重要性，通常设置更大的beta以偏好短距离
        rho=0.5,  # 信息素的蒸发率
        ant_count=100,  # 蚂蚁的数量
        limit=1000,  # 迭代次数
    )

    # 求解TSP问题
    solution = solver.solve(world)

    # 输出最优路径和总距离
    optimal_path = solution.tour
    optimal_distance = solution.distance

    # 找到最长的边
    longest_dist = 0
    longest_index = 0
    for i in range(len(optimal_path) - 1):
        dist = distance_matrix[optimal_path[i], optimal_path[i + 1]]
        if dist > longest_dist:
            longest_dist = dist
            longest_index = i

    # 移除最长边并显示新的旅行顺序
    optimized_path = optimal_path[longest_index + 1:] + optimal_path[:longest_index + 1]
    optimized_distance = optimal_distance - longest_dist
    print("Optimized visit order:", optimized_path)
    print("Optimized visit distance:", optimized_distance)

    total_tour_duration = optimized_distance / speed + df_recommendation['tour_duration'].sum()
    print("Optimized tour duration:", total_tour_duration)

    # 考虑时间约束，根据 day_tour_hours 更新新景点、路线
    # 如果超出 day_tour_hours 小时，则尝试去掉某些景点
    while total_tour_duration > day_tour_hours:
        # 从排名低的景点开始移除
        df_recommendation = df_recommendation[:-1]
        if len(df_recommendation) == 1:
            optimized_route_name = df_recommendation['Attraction']
            total_tour_duration = df_recommendation['tour_duration'].iloc[0]
            hours = int(total_tour_duration)
            minutes = int((total_tour_duration - hours) * 60)
            # 打印最终路线
            st.write("**路线信息：**")
            st.write(optimized_route_name.iloc[0])
            st.write("**预期所用时间：**")
            st.write(f"{hours}小时{minutes}分钟")

            # 初始化'route_oneday_tour.js'
            lng = df_recommendation['lng'].values[0]
            lat = df_recommendation['lat'].values[0]

            data = {
                "lng": lng,
                "lat": lat,
                "Attraction": optimized_route_name
            }
            df = pd.DataFrame(data)

            data_list = df.to_dict(orient='records')

            json_data = json.dumps(data_list, ensure_ascii=False)

            with open('data/map/js/route_oneday_tour.js', 'w', encoding='utf-8') as f:
                f.write(f"var onedayData = {json_data}")

            # 调用'route_oneday_tour.html'画图
            st.components.v1.iframe(
                "https://stevencetanke.github.io/route_recommendations/data/map/html/route_oneday_tour.html",
                height=600)

            return

        # 重新计算路径和总时间
        df_coords = df_attraction.loc[df_recommendation.index]

        # 创建距离矩阵
        distance_matrix = np.zeros((len(df_coords), len(df_coords)))
        for i, (idx1, row1) in enumerate(df_coords.iterrows()):
            for j, (idx2, row2) in enumerate(df_coords.iterrows()):
                distance_matrix[i, j] = haversine(row1['lat'], row1['lng'], row2['lat'], row2['lng'])

        world = pants.World(range(len(distance_matrix)), distance)
        solver = pants.Solver(
            alpha=1.0,  # 信息素重要性
            beta=4.0,  # 启发式信息的重要性，通常设置更大的beta以偏好短距离
            rho=0.5,  # 信息素的蒸发率
            ant_count=100,  # 蚂蚁的数量
            limit=1000,  # 迭代次数
        )

        solution = solver.solve(world)

        optimal_path = solution.tour
        optimal_distance = solution.distance

        longest_dist = 0
        longest_index = 0
        for i in range(len(optimal_path) - 1):
            dist = distance_matrix[optimal_path[i], optimal_path[i + 1]]
            if dist > longest_dist:
                longest_dist = dist
                longest_index = i

        optimized_path = optimal_path[longest_index + 1:] + optimal_path[:longest_index + 1]
        optimized_distance = optimal_distance - longest_dist
        print("Optimized visit order:", optimized_path)
        print("Optimized visit distance:", optimized_distance)

        total_tour_duration = optimized_distance / speed + df_recommendation['tour_duration'].sum()
        print("Optimized tour duration:", total_tour_duration)

    attraction_names = df_coords['Attraction'].tolist()
    optimized_route_names = [attraction_names[idx] for idx in optimized_path]

    # 打印最终路线和预期时间
    st.write("**路线信息：**")
    st.write(" -> ".join(optimized_route_names))

    hours = int(total_tour_duration)
    minutes = int((total_tour_duration - hours) * 60)
    st.write("**预期所用时间：**")
    st.write(f"{hours}小时{minutes}分钟")

    # 初始化'route_oneday_tour.js'
    optimized_coords = [(df_coords.iloc[idx]['lng'], df_coords.iloc[idx]['lat']) for idx in optimized_path]
    lngs, lats = zip(*optimized_coords)

    data = {
        "lng": lngs,
        "lat": lats,
        "Attraction": optimized_route_names
    }
    df = pd.DataFrame(data)

    data_list = df.to_dict(orient='records')

    json_data = json.dumps(data_list, ensure_ascii=False)

    with open('data/map/js/route_oneday_tour.js', 'w', encoding='utf-8') as f:
        f.write(f"var multidayData = {json_data}")

    # 调用'route_oneday_tour.html'画图
    st.components.v1.iframe(
        "https://stevencetanke.github.io/route_recommendations/data/map/html/route_oneday_tour.html", height=600)


def multi_day_tour(multi_day_spots, selected_tags):
    df_attraction = pd.read_excel('data/processed/attractions_information.xlsx')

    # 根据 selected_tags 和 weights 得到排序
    label_weights = {}
    for tag in selected_tags:
        label_weights[tag] = st.session_state['weights'][tag]

    # 执行加权得分计算和排序，并根据 multi_day_spots 选出景点
    df_attraction['Weighted_Score'] = 0
    for label, weight in label_weights.items():
        df_attraction['Weighted_Score'] += df_attraction.get(label, 0) * weight
    df_recommendation = df_attraction.sort_values(by='Weighted_Score', ascending=False).head(multi_day_spots)

    # 输出结果
    print("Ranking of attractions under user-selected tags:")
    print(df_recommendation[['Attraction', 'Weighted_Score']].to_string(index=False))

    # 蚁群算法得到最短路线
    # 加载经纬度数据
    df_coords = df_attraction.loc[df_recommendation.index]  # 只获取推荐景点的坐标

    # 创建距离矩阵
    distance_matrix = np.zeros((len(df_coords), len(df_coords)))
    for i, (idx1, row1) in enumerate(df_coords.iterrows()):
        for j, (idx2, row2) in enumerate(df_coords.iterrows()):
            distance_matrix[i, j] = haversine(row1['lat'], row1['lng'], row2['lat'], row2['lng'])

    # 创建距离矩阵的计算函数
    def distance(i, j):
        return distance_matrix[i][j]

    # 创建世界和蚂蚁
    world = pants.World(range(len(distance_matrix)), distance)
    # 定义蚁群算法的参数
    solver = pants.Solver(
        alpha=1.0,  # 信息素重要性
        beta=4.0,  # 启发式信息的重要性，通常设置更大的beta以偏好短距离
        rho=0.5,  # 信息素的蒸发率
        ant_count=100,  # 蚂蚁的数量
        limit=1000,  # 迭代次数
    )

    # 求解TSP问题
    solution = solver.solve(world)

    # 输出最优路径和总距离
    optimal_path = solution.tour
    optimal_distance = solution.distance
    print("Optimal path:", optimal_path + [optimal_path[0]])
    print("Optimal distance:", optimal_distance)

    # 找到最长的边
    longest_dist = 0
    longest_index = 0
    for i in range(len(optimal_path) - 1):
        dist = distance_matrix[optimal_path[i], optimal_path[i + 1]]
        if dist > longest_dist:
            longest_dist = dist
            longest_index = i

    # 移除最长边并显示新的旅行顺序
    optimized_path = optimal_path[longest_index + 1:] + optimal_path[:longest_index + 1]
    print("Optimized visit order:", optimized_path)
    print("Optimized visit distance:", optimal_distance - longest_dist)

    # 从df_coords获取景点名称
    attraction_names = df_coords['Attraction'].tolist()

    # 生成优化后的景点访问顺序的名称
    optimized_route_names = [attraction_names[idx] for idx in optimized_path]

    # 打印最终路线
    st.write("**路线信息：**")
    st.write(" -> ".join(optimized_route_names))

    # 初始化'route_multiday_tour.js'
    optimized_coords = [(df_coords.iloc[idx]['lng'], df_coords.iloc[idx]['lat']) for idx in optimized_path]
    lngs, lats = zip(*optimized_coords)

    data = {
        "lng": lngs,
        "lat": lats,
        "Attraction": optimized_route_names
    }
    df = pd.DataFrame(data)

    data_list = df.to_dict(orient='records')

    json_data = json.dumps(data_list, ensure_ascii=False)

    with open('data/map/js/route_multiday_tour.js', 'w', encoding='utf-8') as f:
        f.write(f"var multidayData = {json_data}")

    # 调用'route_multiday_tour.html'画图
    st.components.v1.iframe(
        "https://stevencetanke.github.io/route_recommendations/data/map/html/route_multiday_tour.html", height=600)


# 计算地球上两点间的距离
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径（公里）
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def main():
    tour_type, day_tour_hours, multi_day_spots, selected_tags = setup()
    validate_weights_expectation(tour_type, day_tour_hours, multi_day_spots, selected_tags)


if __name__ == "__main__":
    main()
