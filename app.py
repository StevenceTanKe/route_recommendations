import random

import streamlit as st
import pandas as pd
import numpy as np
import pants
import json

from deap import base, creator, tools, algorithms


def setup():
    st.set_page_config(
        page_title='澳门旅游路线推荐系统',
        page_icon='data/map/icon/macau.ico',
        layout='wide'
    )

    st.title('澳门旅游路线推荐系统')
    st.write('根据您的 *旅游类型* 和 *兴趣* 推荐最适合的旅游路线。')

    if 'oneday_tour' not in st.session_state:
        st.session_state['oneday_tour'] = {}
        st.session_state['oneday_tour']['day_tour'] = 1
        st.session_state['oneday_tour']['weights'] = {}
        st.session_state['oneday_tour']['df_weigh'] = None
        st.session_state['oneday_tour']['selected_tags'] = []
        st.session_state['oneday_tour']['route_info'] = None
        st.session_state['oneday_tour']['hours'] = None
        st.session_state['oneday_tour']['minutes'] = None
        st.session_state['oneday_tour']['html_content_with_data'] = None
        st.session_state['oneday_tour']['options'] = []
        st.session_state['oneday_tour']['data_list'] = []

    if 'multiday_tour' not in st.session_state:
        st.session_state['multiday_tour'] = {}
        st.session_state['multiday_tour']['multi_day'] = 2
        st.session_state['multiday_tour']['weights'] = {}
        st.session_state['multiday_tour']['df_weigh'] = None
        st.session_state['multiday_tour']['selected_tags'] = []
        st.session_state['multiday_tour']['route_info'] = None
        st.session_state['multiday_tour']['html_content_with_data'] = None

    if 'loading' not in st.session_state:
        st.session_state.loading = True

    if 'tour_type' not in st.session_state:
        st.session_state['tour_type'] = '一日游'

    st.sidebar.title("旅游路线设置")
    st.session_state['tour_type'] = st.sidebar.radio('请选择旅游类型：', ('一日游', '多日游'))

    if st.session_state['tour_type'] == '一日游':
        day_tour = st.sidebar.number_input(
            '请输入一天期望游览的小时数',
            min_value=1,
            max_value=24,
            help='小时数范围：1~24',
            value=st.session_state['oneday_tour'].get('day_tour', 1),
            key='day_tour')

        selected_tags = st.sidebar.multiselect('请选择你感兴趣的标签：', options=['Square', 'Park', 'Street',
                                                                                 'Museum', 'Performance',
                                                                                 'Western-Church', 'Chinese',
                                                                                 'Landmark'],
                                               default=st.session_state['oneday_tour']['selected_tags'],
                                               key='oneday_tags')

        validate_weights_expectation(selected_tags, day_tour)
    elif st.session_state['tour_type'] == '多日游':
        multi_day = st.sidebar.number_input(
            '请输入期望游览的景点数量',
            min_value=2,
            max_value=49,
            help='景点数量范围：2~49',
            value=st.session_state['multiday_tour'].get('multi_day', 2),
            key='multi_day')

        selected_tags = st.sidebar.multiselect('请选择你感兴趣的标签：', options=['Square', 'Park', 'Street',
                                                                                 'Museum', 'Performance',
                                                                                 'Western-Church',
                                                                                 'Chinese', 'Landmark'],
                                               default=st.session_state['multiday_tour']['selected_tags'],
                                               key='multiday_tags'
                                               )

        validate_weights_expectation(selected_tags, multi_day)


def validate_weights_expectation(selected_tags, number):
    total_weight = 0
    weights = {}

    if st.session_state['tour_type'] == '一日游':
        if selected_tags:
            st.sidebar.write(" ")
            st.sidebar.write(" ")
            st.sidebar.write("**请输入每个标签的权重，权重总和应为 1**")

            for tag in selected_tags:
                weights[tag] = st.sidebar.number_input(
                    f"权重（{tag}）",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state['oneday_tour']['weights'].get(tag, 0.0),
                    step=0.01,
                    key=f'oneday_{tag}')
                total_weight += weights[tag]

            if total_weight == 1.0:
                st.sidebar.success("权重验证成功！")

                data = {'标签': [], '权重': []}
                for tag in selected_tags:
                    data['标签'].append(tag)
                    data['权重'].append(weights[tag])

                st.session_state['oneday_tour']['df_weigh'] = pd.DataFrame(data)
                st.session_state['oneday_tour']['df_weigh'].set_index('标签', inplace=True)  # 将标签列设置为索引

                # 主页面显示
                st.table(st.session_state['oneday_tour']['df_weigh'])

                # 路线信息
                if st.session_state['oneday_tour']['route_info'] is not None:
                    st.write("**路线信息：**")
                    st.write(st.session_state['oneday_tour']['route_info'])

                # 预期所用时间
                if st.session_state['oneday_tour']['hours'] and st.session_state['oneday_tour']['minutes'] is not None:
                    st.write("**预期所用时间：**")
                    st.write(
                        f"{st.session_state['oneday_tour']['hours']}小时{st.session_state['oneday_tour']['minutes']}分钟")

                # 地图展示
                if st.session_state['oneday_tour']['html_content_with_data'] is not None:
                    # 使用Streamlit的components模块显示HTML内容
                    st.components.v1.html(st.session_state['oneday_tour']['html_content_with_data'], height=666)

                # 当只有一个景点时，取消导航功能
                if len(st.session_state['oneday_tour']['data_list']) != 1:
                    # 导航栏展示
                    if st.session_state['oneday_tour']['options']:
                        st.write()
                        selected_option = st.selectbox('请选择您需要导航的路段：',
                                                       st.session_state['oneday_tour']['options'])

                        # 创建一个确认按钮
                        if st.button("确认"):
                            # st.write("你选择了路段:", selected_option) 画出导航地图
                            index = st.session_state['oneday_tour']['options'].index(selected_option)
                            data_list = [st.session_state['oneday_tour']['data_list'][index],
                                         st.session_state['oneday_tour']['data_list'][index + 1]]
                            json_data = json.dumps(data_list, ensure_ascii=False)

                            # 读入基础html页面
                            with open('data/map/html/route_navigation.html', 'r', encoding='utf-8') as f:
                                html_navigation_basic = f.read()

                            html_navigation = f"""
                                        <script>
                                        var navigData = {json_data};
                                        </script>
                                        {html_navigation_basic}
                                        """

                            st.write("**具体路线导航：**")
                            st.components.v1.html(html_navigation, height=666)

                # 当用户点击“路线推荐”按钮
                if st.sidebar.button('路线推荐'):
                    # 将之前显示的推荐路线信息置空
                    st.session_state['oneday_tour']['route_info'] = None
                    st.session_state['oneday_tour']['hours'] = None
                    st.session_state['oneday_tour']['minutes'] = None
                    st.session_state['oneday_tour']['html_content_with_data'] = None

                    # 为了流畅性和绑定数据，保存变量
                    st.session_state['oneday_tour']['day_tour'] = number
                    st.session_state['oneday_tour']['selected_tags'] = selected_tags
                    st.session_state['oneday_tour']['weights'] = weights

                    # st.success(
                    #     f"根据您选择的旅游类型 **{st.session_state['tour_type']}** 和以上标签及权重为您推荐路线...")

                    one_day_tour()
            else:
                st.sidebar.error(f"权重和为 *{total_weight:.2f}*，请调整以满足总和为 *1* 的要求。")
                st.sidebar.button("路线推荐", disabled=True)

    elif st.session_state['tour_type'] == '多日游':
        if selected_tags:
            st.sidebar.write(" ")
            st.sidebar.write(" ")
            st.sidebar.write("**请输入每个标签的权重，权重总和应为 1**")

            for tag in selected_tags:
                weights[tag] = st.sidebar.number_input(
                    f"权重（{tag}）",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state['multiday_tour']['weights'].get(tag, 0.0),
                    step=0.01,
                    key=f'multiday_{tag}')
                total_weight += weights[tag]

            if total_weight == 1.0:
                st.sidebar.success("权重验证成功！")

                data = {'标签': [], '权重': []}
                for tag in selected_tags:
                    data['标签'].append(tag)
                    data['权重'].append(weights[tag])

                st.session_state['multiday_tour']['df_weigh'] = pd.DataFrame(data)
                st.session_state['multiday_tour']['df_weigh'].set_index('标签', inplace=True)  # 将标签列设置为索引

                st.table(st.session_state['multiday_tour']['df_weigh'])

                # 路线信息
                if st.session_state['multiday_tour']['route_info'] is not None:
                    st.write("**路线信息：（仅供参考，您可自主选择起点）**")
                    st.write(st.session_state['multiday_tour']['route_info'])

                # 地图展示
                if st.session_state['multiday_tour']['html_content_with_data'] is not None:
                    # 使用Streamlit的components模块显示HTML内容
                    st.components.v1.html(st.session_state['multiday_tour']['html_content_with_data'], height=666)

                if st.sidebar.button('路线推荐'):
                    st.session_state['multiday_tour']['route_info'] = None
                    st.session_state['multiday_tour']['html_content_with_data'] = None

                    st.session_state['multiday_tour']['multi_day'] = number
                    st.session_state['multiday_tour']['selected_tags'] = selected_tags
                    st.session_state['multiday_tour']['weights'] = weights

                    # st.success(
                    #     f"根据您选择的旅游类型 **{st.session_state['tour_type']}** 和以上标签及权重为您推荐路线...")

                    multi_day_tour()
            else:
                st.sidebar.error(f"权重和为 *{total_weight:.2f}*，请调整以满足总和为 *1* 的要求。")
                st.sidebar.button("路线推荐", disabled=True)


def one_day_tour():
    with st.sidebar.status("路线加载中，请稍后..."):

        st.write("正在加载景点...")

        df_attraction = pd.read_excel('data/processed/attractions_information.xlsx')

        # 根据 selected_tags 和 weights 得到排序
        speed = 5

        label_weights = {}
        for tag in st.session_state['oneday_tour']['selected_tags']:
            label_weights[tag] = st.session_state['oneday_tour']['weights'][tag]

        # 执行加权得分计算和排序
        df_attraction['Weighted_Score'] = 0
        for label, weight in label_weights.items():
            df_attraction['Weighted_Score'] += df_attraction.get(label, 0) * weight
        df_recommendation = df_attraction.sort_values(by='Weighted_Score', ascending=False).head(
            st.session_state['oneday_tour']['day_tour'] + 1)

        print(df_recommendation[['Attraction', 'Weighted_Score']].to_string(index=False))
        print(f"\n Stay duration: {df_recommendation['tour_duration'].sum()} h")

        st.write("🎉景点加载成功")

        st.write("正在加载路线...")

        # 混合遗传算法得到最短路线
        # 加载经纬度数据
        df_coords = df_attraction.loc[df_recommendation.index]  # 只获取推荐景点的坐标

        # 创建距离矩阵
        distance_matrix = np.zeros((len(df_coords), len(df_coords)))
        for i, (idx1, row1) in enumerate(df_coords.iterrows()):
            for j, (idx2, row2) in enumerate(df_coords.iterrows()):
                distance_matrix[i, j] = haversine(row1['lat'], row1['lng'], row2['lat'], row2['lng'])

        # Define the fitness function
        def fitness(individual):
            return (sum(distance_matrix[individual[i], individual[i + 1]] for i in range(len(individual) - 1)),)

        # Set up the genetic algorithm
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, range(len(df_coords)), len(df_coords))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", fitness)

        # Initialize the population
        population = toolbox.population(n=300)

        # Set up the algorithm parameters
        ngen = 400
        cxpb = 0.7
        mutpb = 0.2

        # Run the genetic algorithm
        best_individuals = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

        # Find the optimal solution
        best_individual = tools.selBest(best_individuals[0], k=1)[
            0]  # Note that best_individuals[0] is used here to obtain the final population
        optimal_path = best_individual
        optimal_distance = best_individual.fitness.values[0]  # Use fitness.values to access fitness values

        # 打印结果
        attraction_names = df_coords['Attraction'].tolist()
        optimized_route_names = [attraction_names[idx] for idx in optimal_path]

        total_tour_duration = optimal_distance / speed + df_recommendation['tour_duration'].sum()
        print("Optimized tour duration:", total_tour_duration)

        # 考虑时间约束，根据 day_tour_hours 更新新景点、路线
        # 如果超出 day_tour_hours 小时，则尝试去掉某些景点
        while total_tour_duration > st.session_state['oneday_tour']['day_tour']:
            # 从排名低的景点开始移除
            df_recommendation = df_recommendation[:-1]

            # 加载经纬度数据
            df_coords = df_attraction.loc[df_recommendation.index]  # 只获取推荐景点的坐标

            if len(df_recommendation) == 1:
                st.session_state['oneday_tour']['route_info'] = df_recommendation['Attraction'].iloc[0]

                st.session_state['oneday_tour']['hours'] = int(df_recommendation['tour_duration'].iloc[0])
                st.session_state['oneday_tour']['minutes'] = int(
                    (df_recommendation['tour_duration'].iloc[0] - st.session_state['oneday_tour']['hours']) * 60)

                data = {
                    "lng": df_recommendation['lng'].iloc[0],
                    "lat": df_recommendation['lat'].iloc[0],
                    "Attraction": df_recommendation['Attraction'].iloc[0]
                }
                data_list = [data]

                json_data = json.dumps(data_list, ensure_ascii=False)

                html_file_path = 'data/map/html/route_oneday_tour.html'

                # 读取HTML文件内容
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # 将JSON数据嵌入到HTML内容中
                st.session_state['oneday_tour']['html_content_with_data'] = f"""
                        <script>
                        var onedayData = {json_data};
                        </script>
                        {html_content}
                        """
                st.write("🥳路线加载成功")

                # navigation(df)
                st.session_state['oneday_tour']['data_list'] = data_list

                if st.sidebar.button('路线展示'):
                    pass

                return

            # 创建距离矩阵
            distance_matrix = np.zeros((len(df_coords), len(df_coords)))
            for i, (idx1, row1) in enumerate(df_coords.iterrows()):
                for j, (idx2, row2) in enumerate(df_coords.iterrows()):
                    distance_matrix[i, j] = haversine(row1['lat'], row1['lng'], row2['lat'], row2['lng'])

            # 适应度函数定义
            def fitness(individual):
                return (sum(distance_matrix[individual[i], individual[i + 1]] for i in range(len(individual) - 1)),)

            # 遗传算法设置
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("indices", random.sample, range(len(df_coords)), len(df_coords))
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("mate", tools.cxOrdered)
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("evaluate", fitness)

            # 初始化种群
            population = toolbox.population(n=300)

            # 算法参数
            ngen = 400
            cxpb = 0.7
            mutpb = 0.2

            # 运行遗传算法
            best_individuals = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

            # 找到最优解
            best_individual = tools.selBest(best_individuals[0], k=1)[0]  # 注意这里使用best_individuals[0]获取最终种群
            optimal_path = best_individual
            optimal_distance = best_individual.fitness.values[0]  # 使用fitness.values来访问适应度值

            # 打印结果
            attraction_names = df_coords['Attraction'].tolist()
            optimized_route_names = [attraction_names[idx] for idx in optimal_path]

            total_tour_duration = optimal_distance / speed + df_recommendation['tour_duration'].sum()
            print("Optimized tour duration:", total_tour_duration)

        attraction_names = df_coords['Attraction'].tolist()
        optimized_route_names = [attraction_names[idx] for idx in optimal_path]

        # 打印最终路线和预期时间
        st.session_state['oneday_tour']['route_info'] = " → ".join(optimized_route_names)

        st.session_state['oneday_tour']['hours'] = int(total_tour_duration)
        st.session_state['oneday_tour']['minutes'] = int(
            (total_tour_duration - st.session_state['oneday_tour']['hours']) * 60)

        optimized_coords = [(df_coords.iloc[idx]['lng'], df_coords.iloc[idx]['lat']) for idx in optimal_path]
        lngs, lats = zip(*optimized_coords)
        data = {
            "lng": lngs,
            "lat": lats,
            "Attraction": optimized_route_names
        }
        df = pd.DataFrame(data)

        data_list = df.to_dict(orient='records')

        json_data = json.dumps(data_list, ensure_ascii=False)

        html_file_path = 'data/map/html/route_oneday_tour.html'

        # 读取HTML文件内容
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 将JSON数据嵌入到HTML内容中
        st.session_state['oneday_tour']['html_content_with_data'] = f"""
                    <script>
                    var onedayData = {json_data};
                    </script>
                    {html_content}
                    """
        st.write("🥳路线加载成功")

        # navigation(df)
        st.session_state['oneday_tour']['options'] = [f"{df['Attraction'].iloc[i]} → {df['Attraction'].iloc[i + 1]}" for
                                                      i in range(len(df) - 1)]
        st.session_state['oneday_tour']['data_list'] = data_list

    if st.sidebar.button('路线展示'):
        pass


def multi_day_tour():
    with st.sidebar.status("路线加载中，请稍后..."):

        st.write("正在加载景点...")

        df_attraction = pd.read_excel('data/processed/attractions_information.xlsx')

        # 根据 selected_tags 和 weights 得到排序
        label_weights = {}
        for tag in st.session_state['multiday_tour']['selected_tags']:
            label_weights[tag] = st.session_state['multiday_tour']['weights'][tag]

        # 执行加权得分计算和排序，并根据 multi_day_spots 选出景点
        df_attraction['Weighted_Score'] = 0
        for label, weight in label_weights.items():
            df_attraction['Weighted_Score'] += df_attraction.get(label, 0) * weight
        df_recommendation = df_attraction.sort_values(by='Weighted_Score', ascending=False).head(
            st.session_state['multiday_tour']['multi_day'])

        # 输出结果
        print("Ranking of attractions under user-selected tags:")
        print(df_recommendation[['Attraction', 'Weighted_Score']].to_string(index=False))

        st.write("🎉景点加载成功")

        st.write("正在加载路线...")

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
        optimal_path = optimal_path + [optimal_path[0]]
        optimal_distance = solution.distance
        print("Optimal path:", optimal_path + [optimal_path[0]])
        print("Optimal distance:", optimal_distance)

        # 从df_coords获取景点名称
        attraction_names = df_coords['Attraction'].tolist()

        # 生成优化后的景点访问顺序的名称
        optimized_route_names = [attraction_names[idx] for idx in optimal_path]

        # 打印最终路线
        st.session_state['multiday_tour']['route_info'] = " → ".join(optimized_route_names)

        # 得到json数据
        optimized_coords = [(df_coords.iloc[idx]['lng'], df_coords.iloc[idx]['lat']) for idx in optimal_path]
        lngs, lats = zip(*optimized_coords)

        data = {
            "lng": lngs,
            "lat": lats,
            "Attraction": optimized_route_names
        }
        df = pd.DataFrame(data)

        data_list = df.to_dict(orient='records')

        json_data = json.dumps(data_list, ensure_ascii=False)

        html_file_path = 'data/map/html/route_multiday_tour.html'

        # 读取HTML文件内容
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 将JSON数据嵌入到HTML内容中
        st.session_state['multiday_tour']['html_content_with_data'] = f"""
        <script>
        var multidayData = {json_data};
        </script>
        {html_content}
        """

        st.write("🥳路线加载成功")

    if st.sidebar.button('路线展示'):
        pass


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
    setup()


if __name__ == "__main__":
    main()
