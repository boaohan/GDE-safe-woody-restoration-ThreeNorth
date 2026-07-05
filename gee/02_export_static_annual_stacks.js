/****
02_export_static_annual_stacks.js

Exports the static stack and annual stacks used to aggregate hydroclimatic, vegetation, GDE, GWSA, and wind-erosion metrics by ClassCode.

Organized for repository release from the project process notes. Run in the
Google Earth Engine Code Editor. Update project asset paths in the parameter
section if your GEE asset IDs differ.
****/

/****
Step3_Export_AnnualStacks_ForPython_FULL
三北地区：第三步正式版完整版（可直接运行）
用途：
1) 导出静态栅格堆栈 ThreeNorth_StaticStack_Formal_V2
2) 导出逐年年度影像栈 ThreeNorth_AnnualStack_2005 ~ 2024
3) 同时补入真实生态功能波段：
   NDVI_gs / FVC / NPP / WUE / BareSoilFrac / SandConnectivity / WindErosionRisk
4) 后续在 Python 中按 ClassCode 汇总生成：
   - ThreeNorth_ClassYear_FullMetrics_2005_2024.csv
   - ThreeNorth_Class_HydroSupport_2005_2024.csv
   - ThreeNorth_Class_EcoFunction.csv
注意：
- 直接粘贴到 GEE Code Editor 即可运行
- 若某个资产路径与你实际不一致，只改“参数区”即可
- 先在 Tasks 里运行静态栈，再运行年度栈
****/
// ============================================================================
// 0. 参数区
// ============================================================================
var START_YEAR = 2005;
var END_YEAR   = 2024;
var GROW_START_MONTH = 5;
var GROW_END_MONTH   = 9;
var TARGET_SCALE = 1000;
// 导出开关
var EXPORT_STATIC_STACK = true;
var EXPORT_ANNUAL_STACKS = true;
// 导出目录
var DRIVE_FOLDER = "GEE_Exports";
// ---------- 研究区 ----------
var roi = ee.FeatureCollection(
  "projects/named-defender-476802-p4/assets/sanbeidiqu/TNRBoundary_noregion"
);
// ---------- 资产 ----------
// 文档中的 Step3 使用了这个路径；若你实际放在 sanbeiGDE，请改成你的真实路径
var ASSET_BASEZONE = "projects/named-defender-476802-p4/assets/qgl/ThreeNorth_EcoHydro_Zones_1km";
var ASSET_GDE_UNION = "projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Union_Mask_2006_2024";
var ASSET_GDE_TRAJ  = "projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Trajectory_Code_2006_2024";
var ASSET_GDE_B1 = "projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Binary_P1_2005_2009";
var ASSET_GDE_B2 = "projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Binary_P2_2010_2014";
var ASSET_GDE_B3 = "projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Binary_P3_2015_2019";
var ASSET_GDE_B4 = "projects/named-defender-476802-p4/assets/sanbeiGDE/GDE_Binary_P4_2020_2024";
var ASSET_GWSA_TREND = "projects/named-defender-476802-p4/assets/sanbeiGDE/GWSA_trend_2005_2024";
var ASSET_GWSA_P1 = "projects/named-defender-476802-p4/assets/sanbeiGDE/GWSA_mean_P1_2005_2009";
var ASSET_GWSA_P2 = "projects/named-defender-476802-p4/assets/sanbeiGDE/GWSA_mean_P2_2010_2014";
var ASSET_GWSA_P3 = "projects/named-defender-476802-p4/assets/sanbeiGDE/GWSA_mean_P3_2015_2019";
var ASSET_GWSA_P4 = "projects/named-defender-476802-p4/assets/sanbeiGDE/GWSA_mean_P4_2020_2024";
// NDVI 土壤线与植被线
var NDVI_SOIL = 0.05;
var NDVI_VEG  = 0.80;
// MOD17 NPP 比例因子（通常是 0.0001）
var NPP_SCALE = 0.0001;
Map.centerObject(roi, 5);
// ============================================================================
// 1. 投影与工具函数
// ============================================================================
var startDate = ee.Date.fromYMD(START_YEAR, 1, 1);
var endDate   = ee.Date.fromYMD(END_YEAR, 12, 31);
var climateRef = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .first();
var targetProj = climateRef.select('pr').projection();
var modisLaiProj = ee.ImageCollection("MODIS/061/MCD15A3H")
  .first().select('Lai').projection();
var mod13Proj = ee.ImageCollection("MODIS/061/MOD13Q1")
  .first().select('NDVI').projection();
var mod17Proj = ee.ImageCollection("MODIS/061/MOD17A3HGF")
  .first().select('Npp').projection();
var mcd12Proj = ee.ImageCollection("MODIS/061/MCD12Q1")
  .first().select('LC_Type1').projection();
var srtmProj = ee.Image("USGS/SRTMGL1_003")
  .select('elevation').projection();
var gswProj = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
  .select('occurrence').projection();
var gdeProj = ee.Image(ASSET_GDE_UNION)
  .select(0).projection();
var baseZoneProj = ee.Image(ASSET_BASEZONE)
  .select(0).projection();
function setProj(img, proj) {
  return ee.Image(img).setDefaultProjection(proj);
}
function resampleCoarseTo1km(img, outName) {
  return ee.Image(img)
    .resample('bilinear')
    .reproject({crs: targetProj, scale: TARGET_SCALE})
    .rename(outName)
    .clip(roi);
}
function aggHighResMeanTo1km(img, outName, srcProj) {
  return setProj(img, srcProj)
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 4096
    })
    .reproject({crs: targetProj, scale: TARGET_SCALE})
    .rename(outName)
    .clip(roi);
}
function aggHighResMaxTo1km(img, outName, srcProj) {
  return setProj(img, srcProj)
    .reduceResolution({
      reducer: ee.Reducer.max(),
      maxPixels: 4096
    })
    .reproject({crs: targetProj, scale: TARGET_SCALE})
    .rename(outName)
    .clip(roi);
}
function normalize01(img, minVal, maxVal, outName) {
  return ee.Image(img)
    .subtract(minVal)
    .divide(ee.Number(maxVal).subtract(minVal))
    .clamp(0, 1)
    .rename(outName);
}
// ============================================================================
// 2. BaseZone / GDE / ClassCode
// ============================================================================
var baseZone = ee.Image(ASSET_BASEZONE)
  .select(0)
  .rename('BaseZone')
  .setDefaultProjection(baseZoneProj)
  .toInt16()
  .reproject({crs: targetProj, scale: TARGET_SCALE})
  .clip(roi);
var gdeUnionRaw = ee.Image(ASSET_GDE_UNION)
  .select(0)
  .gt(0)
  .unmask(0)
  .rename('GDE_union_raw')
  .setDefaultProjection(gdeProj)
  .clip(roi);
var gdeFrac = aggHighResMeanTo1km(
  gdeUnionRaw,
  'GDE_frac',
  gdeProj
);
// 0=无约束, 1=核心保育低敏感, 2=中度敏感, 3=高敏感约束
var gdeLevel = ee.Image(0)
  .where(gdeFrac.gte(0.60), 1)
  .where(gdeFrac.gte(0.30).and(gdeFrac.lt(0.60)), 2)
  .where(gdeFrac.gte(0.05).and(gdeFrac.lt(0.30)), 3)
  .toInt8()
  .rename('GDE_Level')
  .clip(roi);
var classCode = baseZone.multiply(10).add(gdeLevel)
  .toInt16()
  .rename('ClassCode')
  .clip(roi);
var gdeTrajectory = ee.Image(ASSET_GDE_TRAJ)
  .select(0)
  .rename('GDE_trajectory_code')
  .setDefaultProjection(gdeProj)
  .toInt16()
  .reproject({crs: targetProj, scale: TARGET_SCALE})
  .clip(roi);
var gdeB1 = ee.Image(ASSET_GDE_B1).select(0).gt(0).unmask(0).setDefaultProjection(gdeProj);
var gdeB2 = ee.Image(ASSET_GDE_B2).select(0).gt(0).unmask(0).setDefaultProjection(gdeProj);
var gdeB3 = ee.Image(ASSET_GDE_B3).select(0).gt(0).unmask(0).setDefaultProjection(gdeProj);
var gdeB4 = ee.Image(ASSET_GDE_B4).select(0).gt(0).unmask(0).setDefaultProjection(gdeProj);
var gdePersistenceRaw = gdeB1.add(gdeB2).add(gdeB3).add(gdeB4)
  .rename('GDE_persistence_raw')
  .setDefaultProjection(gdeProj);
var gdePersistenceCount = aggHighResMeanTo1km(
  gdePersistenceRaw,
  'GDE_persistence_count',
  gdeProj
);
var gdeStability = gdePersistenceCount.divide(4)
  .rename('GDE_stability')
  .clip(roi);
// ============================================================================
// 3. GWSA
// ============================================================================
var gwsaTrend = resampleCoarseTo1km(
  ee.Image(ASSET_GWSA_TREND).select(0),
  'GWSA_trend'
);
var gwsaP1 = resampleCoarseTo1km(
  ee.Image(ASSET_GWSA_P1).select(0),
  'GWSA_mean_period'
);
var gwsaP2 = resampleCoarseTo1km(
  ee.Image(ASSET_GWSA_P2).select(0),
  'GWSA_mean_period'
);
var gwsaP3 = resampleCoarseTo1km(
  ee.Image(ASSET_GWSA_P3).select(0),
  'GWSA_mean_period'
);
var gwsaP4 = resampleCoarseTo1km(
  ee.Image(ASSET_GWSA_P4).select(0),
  'GWSA_mean_period'
);
function getGWSAByYear(year) {
  year = ee.Number(year).int();
  return ee.Image(
    ee.Algorithms.If(
      year.lte(2009), gwsaP1,
      ee.Algorithms.If(
        year.lte(2014), gwsaP2,
        ee.Algorithms.If(
          year.lte(2019), gwsaP3, gwsaP4
        )
      )
    )
  ).rename('GWSA_mean_period');
}
// ============================================================================
// 4. DEM / slope / 地表水
// ============================================================================
var demRaw = ee.Image("USGS/SRTMGL1_003")
  .select('elevation')
  .setDefaultProjection(srtmProj)
  .clip(roi);
var dem = aggHighResMeanTo1km(
  demRaw,
  'Elevation',
  srtmProj
);
var slopeRaw = ee.Terrain.slope(demRaw)
  .rename('Slope_raw')
  .setDefaultProjection(srtmProj);
var slope = aggHighResMeanTo1km(
  slopeRaw,
  'Slope',
  srtmProj
);
var gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(roi);
var swOccurrenceRaw = gsw.select('occurrence')
  .rename('SW_occurrence_raw')
  .setDefaultProjection(gswProj);
var swOccurrence = aggHighResMeanTo1km(
  swOccurrenceRaw,
  'SW_occurrence',
  gswProj
);
var permanentWaterRaw = gsw.select('occurrence')
  .gte(90)
  .selfMask()
  .rename('PermanentWater_raw')
  .setDefaultProjection(gswProj);
var permanentWater1km = aggHighResMaxTo1km(
  permanentWaterRaw,
  'PermanentWater',
  gswProj
).gt(0).rename('PermanentWater');
var distToPermanentWater = permanentWater1km.selfMask()
  .distance(ee.Kernel.euclidean(50000, 'meters'))
  .reproject({crs: targetProj, scale: TARGET_SCALE})
  .rename('Dist_perm_water_m')
  .clip(roi);
var swAccess = ee.Image(1)
  .subtract(distToPermanentWater.divide(50000))
  .clamp(0, 1)
  .rename('SW_access')
  .clip(roi);
// ============================================================================
// 5. 自然植被掩膜
// ============================================================================
var lcCol = ee.ImageCollection("MODIS/061/MCD12Q1")
  .filterDate('2018-01-01', '2022-12-31')
  .select('LC_Type1');
var lcMode = lcCol.mode()
  .rename('LC_Type1_mode')
  .setDefaultProjection(mcd12Proj)
  .clip(roi);
var naturalMask500m = lcMode.gte(1).and(lcMode.lte(11))
  .rename('NaturalMask_500m')
  .setDefaultProjection(mcd12Proj);
var naturalFrac1km = aggHighResMeanTo1km(
  naturalMask500m,
  'NaturalFrac',
  mcd12Proj
);
var naturalMask1km = naturalFrac1km.gte(0.5).rename('NaturalMask_1km');
// ============================================================================
// 6. 当前状态 LAI
// ============================================================================
function getLaiCollectionByRange(startY, endY) {
  return ee.ImageCollection("MODIS/061/MCD15A3H")
    .filterBounds(roi)
    .filterDate(ee.Date.fromYMD(startY, 1, 1), ee.Date.fromYMD(endY, 12, 31))
    .filter(ee.Filter.calendarRange(GROW_START_MONTH, GROW_END_MONTH, 'month'))
    .select('Lai')
    .map(function(img) {
      return ee.Image(img)
        .multiply(0.1)
        .rename('Lai')
        .setDefaultProjection(modisLaiProj);
    });
}
var currentStartYear = END_YEAR - 2;
var laiCurrentRaw = getLaiCollectionByRange(currentStartYear, END_YEAR)
  .mean()
  .rename('LAI_current_raw')
  .setDefaultProjection(modisLaiProj);
var laiCurrent3yr = aggHighResMeanTo1km(
  laiCurrentRaw,
  'LAI_current_3yr',
  modisLaiProj
);
// ============================================================================
// 7. 生态功能辅助函数
// ============================================================================
// 7.1 生长季 NDVI / FVC / BareSoilFrac / SandConnectivity
function getGrowingSeasonEco(year) {
  var yStart = ee.Date.fromYMD(year, 1, 1);
  var yEnd   = yStart.advance(1, 'year');
  var ndviCol = ee.ImageCollection("MODIS/061/MOD13Q1")
    .filterBounds(roi)
    .filterDate(yStart, yEnd)
    .filter(ee.Filter.calendarRange(GROW_START_MONTH, GROW_END_MONTH, 'month'))
    .select('NDVI')
    .map(function(img) {
      return ee.Image(img)
        .multiply(0.0001)
        .rename('NDVI')
        .setDefaultProjection(mod13Proj);
    });
  var ndviMeanRaw = ndviCol.mean()
    .rename('NDVI_gs_raw')
    .setDefaultProjection(mod13Proj);
  var ndviMean = aggHighResMeanTo1km(
    ndviMeanRaw,
    'NDVI_gs',
    mod13Proj
  );
  var fvcRaw = ndviMeanRaw
    .subtract(NDVI_SOIL)
    .divide(NDVI_VEG - NDVI_SOIL)
    .clamp(0, 1)
    .rename('FVC_raw')
    .setDefaultProjection(mod13Proj);
  var fvc = aggHighResMeanTo1km(
    fvcRaw,
    'FVC',
    mod13Proj
  );
  var bareMaskRaw = fvcRaw.lt(0.15)
    .rename('BareMask_raw')
    .setDefaultProjection(mod13Proj);
  var bareFrac = aggHighResMeanTo1km(
    bareMaskRaw,
    'BareSoilFrac',
    mod13Proj
  );
  var sandConnRaw = bareMaskRaw
    .focal_mean({radius: 3, units: 'pixels'})
    .rename('SandConnectivity_raw')
    .setDefaultProjection(mod13Proj);
  var sandConn = aggHighResMeanTo1km(
    sandConnRaw,
    'SandConnectivity',
    mod13Proj
  );
  return {
    ndvi: ndviMean,
    fvc: fvc,
    bare: bareFrac,
    sandConn: sandConn
  };
}
// 7.2 年 NPP
function getAnnualNPP(year) {
  var yStart = ee.Date.fromYMD(year, 1, 1);
  var yEnd   = yStart.advance(1, 'year');
  var nppImg = ee.ImageCollection("MODIS/061/MOD17A3HGF")
    .filterBounds(roi)
    .filterDate(yStart, yEnd)
    .select('Npp')
    .first();
  var nppRaw = ee.Image(nppImg)
    .multiply(NPP_SCALE)
    .rename('NPP_raw')
    .setDefaultProjection(mod17Proj);
  return aggHighResMeanTo1km(
    nppRaw,
    'NPP',
    mod17Proj
  );
}
// 7.3 春季风速（3-5 月）
function getSpringWindSpeed(year) {
  var yStart = ee.Date.fromYMD(year, 1, 1);
  var yEnd   = yStart.advance(1, 'year');
  var era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
    .filterBounds(roi)
    .filterDate(yStart, yEnd)
    .filter(ee.Filter.calendarRange(3, 5, 'month'));
  var windRaw = era.map(function(img) {
    var u = ee.Image(img).select('u_component_of_wind_10m');
    var v = ee.Image(img).select('v_component_of_wind_10m');
    return u.pow(2).add(v.pow(2)).sqrt().rename('wind10');
  }).mean();
  return resampleCoarseTo1km(
    windRaw,
    'Wind10_spring'
  );
}
// ============================================================================
// 8. 年尺度影像
// ============================================================================
function makeAnnualImage(year) {
  year = ee.Number(year).int();
  var yStart = ee.Date.fromYMD(year, 1, 1);
  var yEnd   = yStart.advance(1, 'year');
  // ----------------------------
  // 8.1 TerraClimate 年尺度水文气象
  // ----------------------------
  var tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
    .filterBounds(roi)
    .filterDate(yStart, yEnd);
  var pr = resampleCoarseTo1km(tc.select('pr').sum(), 'P');
  var pet = resampleCoarseTo1km(tc.select('pet').sum(), 'PET');
  var aet = resampleCoarseTo1km(tc.select('aet').sum(), 'AET');
  var runoff = resampleCoarseTo1km(tc.select('ro').sum(), 'Runoff');
  var soil = resampleCoarseTo1km(tc.select('soil').mean(), 'Soil');
  var tmeanRaw = tc.map(function(img) {
    return img.select('tmmx')
      .add(img.select('tmmn'))
      .divide(20)
      .rename('Tmean');
  }).mean();
  var tmean = resampleCoarseTo1km(tmeanRaw, 'Tmean');
  // ----------------------------
  // 8.2 GWSA / AI
  // ----------------------------
  var gwsaMeanPeriod = getGWSAByYear(year);
  var aiPrelim = pr
    .add(soil)
    .add(runoff)
    .divide(pet.max(1))
    .rename('AI_prelim_noGW');
  var aiWithGWSAProxy = pr
    .add(soil)
    .add(runoff)
    .add(gwsaMeanPeriod.unitScale(-50, 50))
    .divide(pet.max(1))
    .rename('AI_with_GWSA_proxy');
  // ----------------------------
  // 8.3 LAI
  // ----------------------------
  var laiYearCol = ee.ImageCollection("MODIS/061/MCD15A3H")
    .filterBounds(roi)
    .filterDate(yStart, yEnd)
    .filter(ee.Filter.calendarRange(GROW_START_MONTH, GROW_END_MONTH, 'month'))
    .select('Lai')
    .map(function(img) {
      return ee.Image(img)
        .multiply(0.1)
        .rename('Lai')
        .setDefaultProjection(modisLaiProj);
    });
  var laiMeanRaw = laiYearCol.mean()
    .rename('LAI_mean_raw')
    .setDefaultProjection(modisLaiProj);
  var laiMaxRaw = laiYearCol.max()
    .rename('LAI_max_raw')
    .setDefaultProjection(modisLaiProj);
  var laiMean = aggHighResMeanTo1km(
    laiMeanRaw,
    'LAI_mean',
    modisLaiProj
  );
  var laiMax = aggHighResMeanTo1km(
    laiMaxRaw,
    'LAI_max',
    modisLaiProj
  );
  var laiNatMean = laiMean.updateMask(naturalMask1km).rename('LAI_nat_mean');
  var laiNatMax  = laiMax.updateMask(naturalMask1km).rename('LAI_nat_max');
  // ----------------------------
  // 8.4 生态功能指标
  // ----------------------------
  var eco = getGrowingSeasonEco(year);
  var ndvi = eco.ndvi;
  var fvc = eco.fvc;
  var bare = eco.bare;
  var sandConn = eco.sandConn;
  var npp = getAnnualNPP(year);
  var wue = npp.divide(aet.max(1e-6)).rename('WUE');
  var wind10 = getSpringWindSpeed(year);
  var windNorm = normalize01(wind10, 0, 12, 'Wind10_norm');
  var windErosionRisk = windNorm.pow(3)
    .multiply(bare)
    .multiply(sandConn)
    .multiply(ee.Image(1).subtract(fvc))
    .rename('WindErosionRisk');
  // ----------------------------
  // 8.5 合成年度栈
  // ----------------------------
  return ee.Image.cat([
    // 水文气象
    pr, pet, aet, runoff, soil, tmean,
    aiPrelim, aiWithGWSAProxy,
    gwsaMeanPeriod,
    // 承载力
    laiMean, laiMax, laiNatMean, laiNatMax,
    // 生态功能
    ndvi, fvc, npp, wue,
    bare, sandConn, wind10, windErosionRisk,
    // 静态背景
    swOccurrence, swAccess,
    dem, slope,
    gdeFrac, gdeLevel, baseZone, classCode,
    gdeStability, gdePersistenceCount, gdeTrajectory,
    gwsaTrend
  ])
  .set('year', year)
  .set('system:time_start', yStart.millis())
  .clip(roi);
}
// ============================================================================
// 9. 静态栈
// ============================================================================
var staticStack = ee.Image.cat([
  baseZone,
  gdeLevel,
  classCode,
  gdeFrac,
  gdeStability,
  gdePersistenceCount,
  gdeTrajectory,
  gwsaTrend,
  dem,
  slope,
  swOccurrence,
  swAccess,
  naturalFrac1km,
  naturalMask1km,
  laiCurrent3yr
]).clip(roi);
// ============================================================================
// 10. 导出
// ============================================================================
// 10.1 静态栈
if (EXPORT_STATIC_STACK) {
  Export.image.toDrive({
    image: staticStack.toFloat(),
    description: 'ThreeNorth_StaticStack_Formal_V2',
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'ThreeNorth_StaticStack_Formal_V2',
    region: roi.geometry(),
    scale: TARGET_SCALE,
    maxPixels: 1e13
  });
}
// 10.2 年度栈
if (EXPORT_ANNUAL_STACKS) {
  var yearList = ee.List.sequence(START_YEAR, END_YEAR).getInfo();
  yearList.forEach(function(y) {
    var annual = makeAnnualImage(y).toFloat();
    Export.image.toDrive({
      image: annual,
      description: 'ThreeNorth_AnnualStack_' + y,
      folder: DRIVE_FOLDER,
      fileNamePrefix: 'ThreeNorth_AnnualStack_' + y,
      region: roi.geometry(),
      scale: TARGET_SCALE,
      maxPixels: 1e13
    });
  });
}
print('Step3 完整版任务已生成。请到 Tasks 面板依次点击 Run。');
print('静态栈：ThreeNorth_StaticStack_Formal_V2');
print('年度栈：ThreeNorth_AnnualStack_2005 ~ ' + END_YEAR);
