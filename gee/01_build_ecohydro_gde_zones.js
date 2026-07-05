/****
01_build_ecohydro_gde_zones.js

Builds the Three-North ecohydrological zones, overlays GDE fractions, and exports the 1-km ClassCode layer.

Organized for repository release from the project process notes. Run in the
Google Earth Engine Code Editor. Update project asset paths in the parameter
section if your GEE asset IDs differ.
****/

// 第一步：构建三北生态水文分区单元 (基础生境聚类)
// 1. 定义研究区 (ROI)
// 请将下方的路径替换为你 GEE Assets 中实际的三北 shp 路径
var roi = ee.FeatureCollection("projects/named-defender-476802-p4/assets/sanbeidiqu/TNRBoundary_noregion");
Map.centerObject(roi, 5);
Map.addLayer(roi, {color: 'red'}, '三北地区边界', false);
// 2. 获取多源生境底图数据 (时间窗口对齐你的 2005-2024 研究期)
var startDate = '2005-01-01';
var endDate = '2024-12-31';
// 2.1 气象与水文数据 (使用 TerraClimate 包含降水、气温、潜在蒸散)
var climate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .select(['pr', 'tmmx', 'pet']) // 降水, 最高气温, 潜在蒸散
  .mean()
  .clip(roi);
// 2.2 地形数据 (使用 SRTM DEM)
var dem = ee.Image("USGS/SRTMGL1_003").clip(roi);
var elevation = dem.select('elevation');
var slope = ee.Terrain.slope(elevation).rename('slope');
// 2.3 组合所有生境变量形成多波段影像
var habitatImage = climate.addBands([elevation, slope]);
var bandNames = habitatImage.bandNames();
// 3. 数据标准化 (Min-Max Normalization)
// 聚类前必须消除不同量纲的影响（如降水几百毫米 vs 坡度几十度）
var minMax = habitatImage.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi.geometry(),
  scale: 5000, // 降采样以加速计算，三北尺度 5km 足够宏观摸底
  maxPixels: 1e13
});
// 构建标准化函数
var normalize = function(image) {
  var mins = ee.Image.constant(minMax.select(bandNames.map(function(n) { return ee.String(n).cat('_min'); })).values());
  var maxs = ee.Image.constant(minMax.select(bandNames.map(function(n) { return ee.String(n).cat('_max'); })).values());
  return image.subtract(mins).divide(maxs.subtract(mins));
};
var normalizedHabitat = normalize(habitatImage);
// 4. 空间无监督聚类 (K-Means 划分基础生境区)
// 提取训练样本 (在 ROI 内随机撒点)
var trainingData = normalizedHabitat.sample({
  region: roi,
  scale: 5000,
  numPixels: 5000, // 采样 5000 个像元用于训练聚类器
  seed: 42
});
// 设定聚类簇数 (例如分为 15 个基础生态水文单元，可根据实际碎化程度调整)
var numClusters = 15;
var clusterer = ee.Clusterer.wekaKMeans(numClusters).train(trainingData);
// 应用聚类
var habitatZones = normalizedHabitat.cluster(clusterer);
// ====== 修复部分开始 ======
// 提取气象数据中“降水(pr)”波段的投影作为全图统一的基准投影
var targetCRS = climate.select('pr').projection();
// 平滑去噪 (消除孤立的单像元噪点，使图斑更具工程实施意义)
var smoothZones = habitatZones.focal_mode({radius: 2, iterations: 1})
                              .reproject({
                                crs: targetCRS, // 【关键修复】：指定单一的明确投影
                                scale: 1000     // 统一重采样到 1km 分辨率
                              })
                              .clip(roi);
// ====== 修复部分结束 ======
// 可视化基础分区结果
var clusterVis = {min: 0, max: numClusters - 1, palette: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']};
Map.addLayer(smoothZones, clusterVis, '基础相似生境分区 (K-Means)');
// ==============================================================================
// 5. 叠置 GDE 敏感性约束 (直接替换上一版代码的预留接口部分)
// ==============================================================================
// 5.1 引入 P1-P4 的 GDE 二值化图层 (1 为 GDE, 0 为非 GDE)
// 请确认路径前缀是否与你的资产路径完全一致
var gde_p1 = ee.Image("projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Binary_P1_2005_2009");
var gde_p2 = ee.Image("projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Binary_P2_2010_2014");
var gde_p3 = ee.Image("projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Binary_P3_2015_2019");
var gde_p4 = ee.Image("projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Binary_P4_2020_2024");
// 计算 20 年间的 GDE 全集掩膜 (Union Mask)
// 只要这 20 年间出现过 GDE，像元值就是 1，否则为 0
// 使用 unmask(0) 确保即使某期数据在局部有空洞，加法依然能正常计算
var gde_union = gde_p1.unmask(0)
  .add(gde_p2.unmask(0))
  .add(gde_p3.unmask(0))
  .add(gde_p4.unmask(0))
  .gt(0);
// 5.2 【核心突破】计算 1km 网格内的 GDE 面积占比 (0.0 ~ 1.0)
// 使用 targetCRS 确保与之前第 4 步基础生境分区的投影和网格完全对齐
var gde_fraction_1km = gde_union
  .reduceResolution({
    reducer: ee.Reducer.mean(), // 计算 1km 网格内值为 1 的原生像元比例
    maxPixels: 1024
  })
  .reproject({
    crs: targetCRS,
    scale: 1000
  });
// 5.3 依据具体的面积占比指标，重新划定硬核敏感性等级 (整数)
// 逻辑定义 (不使用模糊形容词，用具体占比约束)：
// 占比 < 5%   -> 0 (非 GDE 约束区，受影响极小)
// 5% - 30%    -> 3 (高敏感约束：GDE 面积小且极度破碎，极易被人工乔木挤占水分)
// 30% - 60%   -> 2 (中度敏感约束)
// >= 60%      -> 1 (低敏感/核心保育区：GDE 占主导，地下水支撑相对稳定)
var gde_sensitivity = ee.Image(0)
  .where(gde_fraction_1km.gte(0.05).and(gde_fraction_1km.lt(0.3)), 3)
  .where(gde_fraction_1km.gte(0.3).and(gde_fraction_1km.lt(0.6)), 2)
  .where(gde_fraction_1km.gte(0.6), 1);
// 5.4 执行矩阵十进制叠置运算
// 因为 gde_sensitivity 现在是 0,1,2,3 的整数，这里的叠置将完美生成 120, 123 这样的类目编码
var finalEcoHydroZones = smoothZones.multiply(10).add(gde_sensitivity).toInt();
// 5.5 可视化与导出
Map.addLayer(gde_fraction_1km.clip(roi), {min: 0, max: 1, palette: ['white', 'blue']}, '1km网格 GDE 覆盖丰度', false);
Map.addLayer(finalEcoHydroZones.randomVisualizer(), {}, '生态水文联合分区 (含GDE占比约束)');
Export.image.toDrive({
  image: finalEcoHydroZones,
  description: 'ThreeNorth_EcoHydro_Zones_Fraction_Constrained',
  folder: 'GEE_Exports',
  scale: 1000,
  region: roi,
  maxPixels: 1e13
});
// ==============================================================================
// 6. 导出最终生态水文联合分区至 GEE Assets (资产)
// ==============================================================================
Export.image.toAsset({
  image: finalEcoHydroZones.toInt16(), // 强制转为16位整型，既能保存上百的编码，又能大幅节省资产空间
  description: 'Export_EcoHydro_Zones_1km_To_Asset',
  // 这里的路径指向你的 sanbeiGDE 文件夹，你可以根据需要修改最后的文件名
  assetId: 'projects/named-defender-476802-p4/assets/sanbeiGDE/ThreeNorth_EcoHydro_Zones_1km',
  region: roi.geometry(), // 严格限定在三北 ROI 边界内
  scale: 1000,            // 保持 1km (1000米) 的标准网格分辨率
  maxPixels: 1e13,
  // 【关键防错设置】这是分类编码图，缩放金字塔策略必须用 'mode' (众数)
  pyramidingPolicy: {'.default': 'mode'}
});
print('资产导出任务已生成，请去右侧 Tasks 面板点击 Run 执行。');
var numClusters = 15;
var clusterIndices = ee.List.sequence(0, numClusters - 1);
// 提取你需要统计的三个目标波段
var targetBands = habitatImage.select(['pr', 'tmmx', 'elevation']);
// 遍历每一个聚类 ID，分别计算均值
var statsList = clusterIndices.map(function(id) {
  var clusterId = ee.Number(id);
  // 核心：为你当前的类目生成掩膜 (例如只保留分类为 0 的像元)
  var mask = smoothZones.eq(clusterId);
  // 用掩膜过滤气象地形影像，并计算均值
  var meanStats = targetBands.updateMask(mask).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi.geometry(),
    scale: 5000, // 5km 分辨率
    maxPixels: 1e13
  });
  // 组装成带有属性的矢量要素 (Feature)
  return ee.Feature(null, {
    'Cluster_ID': clusterId,
    'Precipitation_mm': meanStats.get('pr'),
    'Temp_Max_C': meanStats.get('tmmx'),
    'Elevation_m': meanStats.get('elevation')
  });
});
// 转换为 FeatureCollection 并导出
var fcStats = ee.FeatureCollection(statsList);
Export.table.toDrive({
  collection: fcStats,
  description: 'ThreeNorth_15Clusters_Stats_Fixed',
  folder: 'GEE_Exports',
  fileFormat: 'CSV'
});
print('修复版表格导出任务已生成，请去 Tasks 面板运行。');
