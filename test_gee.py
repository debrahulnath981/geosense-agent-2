import ee

# Initialize Google Earth Engine
ee.Initialize(project="core-guard-499011-e8")

print("Earth Engine initialized successfully!")

# Chennai point
chennai = ee.Geometry.Point([80.2707, 13.0827])

# Sentinel-2 image collection
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(chennai)
    .filterDate("2023-01-01", "2023-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

# Get the first available image
image = collection.first()

print("Image ID:")
print(image.get("system:index").getInfo())

print("Band list:")
print(image.bandNames().getInfo())