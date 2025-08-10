import ee

# =============================================================================
# --- 1. Initialization and Configuration ---
# =============================================================================
try:
    ee.Initialize(project="deforestationsentinel2")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="deforestationsentinel2")

print("GEE Initialized for FULL DATASET EXPORT.")

# --- Production Parameters ---
# The full Area of Interest for your project.
roi = ee.Geometry.BBox(-74.88851, 1.722, -73.656, 2.712)
roi_area = roi.area(maxError=1).divide(1e6).getInfo()  # Area in km²
print(f"ROI area: {roi_area:.2f} km²")
if roi_area > 10000:
    print("Warning: Large ROI may exceed maxPixels or take significant time.")

# Using a full year provides the best data for the median composite.
start_date = '2022-01-01'
end_date = '2022-12-31'

# Feature bands and export settings
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
LABEL_BAND = 'label'
DRIVE_FOLDER = 'GEE_Colombia_Production_Export'
FILE_PREFIX = 'colombia_production_data'
TASK_DESCRIPTION = 'ProductionExport_Colombia_to_Drive'
PATCH_SIZE = 256
EXPORT_SCALE = 10

# =============================================================================
# --- 2. Data Preparation with Robust Cloud Masking ---
# =============================================================================

def mask_s2_clouds(image):
    """Masks clouds and shadows in a Sentinel-2 SR image using the SCL band."""
    scl = image.select('SCL')
    # Keep high-quality land and water pixels.
    good_quality = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(good_quality)

print("Preparing Sentinel-2 cloud-free composite...")
# Load S2 collection, efficiently filter by metadata first, then map the cloud mask.
s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))) # A lenient filter to get more data
s2_masked = s2_collection.map(mask_s2_clouds)

# Create the median composite from the cleaned collection.
s2_composite = s2_masked.select(S2_BANDS).median()
# Normalize the final feature image.
feature_image = s2_composite.divide(3000).toFloat()

print("Preparing Dynamic World label image...")
# Create the label image using the mode reducer.
dw_collection = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                 .filterBounds(roi)
                 .filterDate(start_date, end_date))
dw_image = dw_collection.select(LABEL_BAND).mode()
label_image = dw_image.toByte()

# Stack the feature and label images for export.
stacked_image = feature_image.addBands(label_image)
print("Data preparation complete. Image stack created.")

# =============================================================================
# --- 3. Export to Google Drive ---
# =============================================================================

print("\nSubmitting the export task...")
task = ee.batch.Export.image.toDrive(
    image=stacked_image,
    description=TASK_DESCRIPTION,
    folder=DRIVE_FOLDER,
    fileNamePrefix=FILE_PREFIX,
    region=roi,
    scale=EXPORT_SCALE,
    fileFormat='TFRecord',
    formatOptions={'patchDimensions': [PATCH_SIZE, PATCH_SIZE], 'compressed': True},
    maxPixels=1e13
)

task.start()

print("\n============================================================")
print("SUCCESS: Full dataset export task has been started.")
print("--> Go to the 'Tasks' tab in the GEE Code Editor to monitor progress.")
print("--> This is a large task and may take several hours to complete.")
print(f"--> When finished, files will appear in Google Drive under the '{DRIVE_FOLDER}' folder.")
print("============================================================")