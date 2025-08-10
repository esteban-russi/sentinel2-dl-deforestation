import ee

# =============================================================================
# --- 1. Initialization and Configuration ---
# =============================================================================
try:
    ee.Initialize(project="deforestationsentinel2")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="deforestationsentinel2")

print("GEE Initialized for CHANGE DETECTION DATA EXPORT.")

# --- General Parameters ---
# The Area of Interest for the change detection analysis.
roi = ee.Geometry.BBox(-72.8, 2.1, -72.2, 2.7) # Corrected BBox order: minX, minY, maxX, maxY
roi_area = roi.area(maxError=1).divide(1e6).getInfo()
print(f"ROI area: {roi_area:.2f} km²")

# --- Export settings ---
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
DRIVE_FOLDER = 'GEE_Colombia_ChangeDetection_Export'
PATCH_SIZE = 256
EXPORT_SCALE = 10

# =============================================================================
# --- 2. Reusable Data Preparation & Export Function ---
# =============================================================================

def mask_s2_clouds(image):
    """Masks clouds and shadows in a Sentinel-2 SR image using the SCL band."""
    scl = image.select('SCL')
    good_quality = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(good_quality)

def export_yearly_composite(year):
    """
    Prepares a cloud-free Sentinel-2 composite for a given year and
    submits an export task to Google Drive.
    """
    print("\n" + "="*50)
    print(f"--- Processing data for the year: {year} ---")
    
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    print("Preparing Sentinel-2 cloud-free composite...")
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(roi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))) # Slightly more lenient for cloudy years
    
    s2_masked = s2_collection.map(mask_s2_clouds)
    s2_composite = s2_masked.select(S2_BANDS).median()
    feature_image = s2_composite.divide(3000).toFloat()
    
    print("Data preparation complete.")

    # --- Configure and start the export task ---
    file_prefix = f'inference_data_{year}'
    task_description = f'ChangeDetectionExport_{year}'
    
    print(f"Submitting export task: '{task_description}'...")
    task = ee.batch.Export.image.toDrive(
        image=feature_image,
        description=task_description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_prefix,
        region=roi,
        scale=EXPORT_SCALE,
        fileFormat='TFRecord',
        formatOptions={'patchDimensions': [PATCH_SIZE, PATCH_SIZE], 'compressed': True},
        maxPixels=1e13
    )
    task.start()
    
    print(f"SUCCESS: Export task for {year} has been started.")
    print("="*50)

# =============================================================================
# --- 3. Main Execution Block ---
# =============================================================================

# --- Define the years you want to compare ---
YEAR_T1 = 2021  # The "before" image
YEAR_T2 = 2023  # The "after" image

# --- Run the export function for each year ---
export_yearly_composite(YEAR_T1)
export_yearly_composite(YEAR_T2)

print("\n\n============================================================")
print("All export tasks have been submitted.")
print("--> Go to the 'Tasks' tab in the GEE Code Editor to monitor progress.")
print(f"--> You should see two new tasks: 'ChangeDetectionExport_{YEAR_T1}' and 'ChangeDetectionExport_{YEAR_T2}'.")
print(f"--> When finished, files will appear in Google Drive under the '{DRIVE_FOLDER}' folder.")
print("============================================================")