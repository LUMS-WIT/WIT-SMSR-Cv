# Load the `terra` package
library(terra)
library(stringr)
library(lubridate)
library(ncdf4)

indir = "data/SMAP-HB_1km_6h/"
outdir = "data/SMAP-HB_1km_CV_daily/"
  
file_paths = list.files(path = indir, pattern = "\\.nc$", full.names = TRUE)

file_paths

shapefile <- vect("CV_shapefile/CV_polygon.shp")
plot(shapefile)

for (file_ in file_paths){
  # message(file_)
  
  # Extract YYYYMM using regex
  yyyymm <- sub(".*_(\\d{6})\\.nc$", "\\1", file_)
  
  dir_name <- file.path(outdir, yyyymm)
  dir.create(dir_name, recursive = TRUE, showWarnings = FALSE)
  
  message("Created folder: ", dir_name)
  
  # Load NC one by one
  # === Load NetCDF as SpatRaster ===
  nc_file <- file_
  
  # Load all variables
  r <- rast(nc_file)
  
  shapefile <- project(shapefile, r)
  
  # === Extract time in seconds from layer names ===
  layer_names <- names(r)
  seconds <- as.numeric(str_extract(layer_names, "(?<=t=)\\d+"))
  
  # === Read time units from NetCDF to get origin ===
  nc <- nc_open(nc_file)
  time_units <- nc$dim$t$units  # or nc$var$time$units if needed
  nc_close(nc)
  
  # === Parse origin from time units (e.g., "seconds since 2015-03-01 00:00:00") ===
  origin_str <- sub(".*since ", "", time_units)
  origin <- ymd_hms(origin_str, tz = "UTC")
  
  datetimes <- origin + seconds
  
  # Group by calendar day
  dates <- as.Date(datetimes)
  unique_dates <- unique(dates)
  unique_dates <- as.Date(unique_dates, origin = origin)
  
  
  # === Loop through each date ===
  for (date in unique_dates) {
    date_format <- as.Date(date, format = "%Y-%m-%d")
    date_format <- format(date_format, "%Y%m%d" )
    message("Processing: ", date_format)
    
    idx <- which(dates == date)
    r_day <- r[[idx]]
    
    # Early skip if all NA before cropping
    if (all(sapply(1:nlyr(r_day), function(i) all(is.na(values(r_day[[i]])))))) {
      message("  Skipping ", date_format, ": all values NA before cropping")
      next
    }
    
    # Crop and mask
    r_day_cropped <- crop(r_day, shapefile)
    r_day_masked <- mask(r_day_cropped, shapefile)
    
    # Optional skip after mask
    if (all(is.na(values(r_day_masked)))) {
      message("  Skipping ", date_format, ": all values NA after masking")
      next
    }
    
    # Pixel-wise mean
    r_day_mean <- app(r_day_masked, fun = mean, na.rm = TRUE)
    
    # Save GeoTIFF
    out_path <- file.path(dir_name, paste0("SMAP-HB_1km_daily_mean_", date_format, ".tif"))
    writeRaster(r_day_mean, out_path, overwrite = TRUE)
    message("  Saved: ", out_path)
  }
  
  
}
  
