library(terra)
library(stringr)
library(fs)  # for file path handling

# Set input and output root folders
in_root <- "data/SMAP-HB_1km_CV_daily"
out_root <- "data/SMAP-HB_3km_CV_daily"

scale_factor <- 3  # e.g., 3 for 3km, 9 for 9km, etc.
coarse_res <- '3km'

# List all .tif files recursively
tif_files <- list.files(in_root, pattern = "\\.tif$", recursive = TRUE, full.names = TRUE)

tif_files

for (file in tif_files) {
  message("Processing: ", file)
  
  # Load fine-resolution raster
  r_fine <- rast(file)
  
  # Get resolution, extent, and CRS of fine raster
  res_fine <- res(r_fine)
  ext_fine <- ext(r_fine)
  crs_fine <- crs(r_fine)
  
  # Compute coarse resolution
  res_coarse <- res_fine * scale_factor
  
  # Compute aligned extent to ensure clean block alignment
  ncol_coarse <- floor((ext_fine[2] - ext_fine[1]) / res_coarse[1])
  nrow_coarse <- floor((ext_fine[4] - ext_fine[3]) / res_coarse[2])
  
  
  xmin = ext_fine[1]
  xmax = ext_fine[1] + ncol_coarse * res_coarse[1]
  ymin = ext_fine[3]
  ymax = ext_fine[3] + nrow_coarse * res_coarse[2]

  aligned_extent <- ext(xmin, xmax, ymin, ymax)

    # Build coarse-resolution template
  r_coarse_template <- rast(
    ext = aligned_extent,
    resolution = res_coarse,
    crs = crs_fine
  )
  
  # Resample fine → coarse
  r_coarse <- resample(r_fine, r_coarse_template, method = "bilinear")
  
  # Output path logic: change "fine" to "coarse"
  rel_path <- path_rel(file, start = in_root)
  rel_out_file <- str_replace(rel_path, "1km", coarse_res)
  out_file <- file.path(out_root, rel_out_file)
  
  dir_create(dirname(out_file))
  writeRaster(r_coarse, out_file, overwrite = TRUE)
  message("Saved: ", out_file)
}


# # === Load NetCDF as SpatRaster ===
# raster_file <- "CV/201503/SMAP-HB_1km_daily_mean_20150331.tif"
# 
# # Load all variables
# r <- rast(raster_file)
# 
# plot(r)
# print(r)
# print(res(r))
# 
# r_1km = r
# res_1km <- res(r_1km)
# res_9km <- res_1km * 9
# 
# r_9km_template <- rast(
#   extent = ext(r_1km),
#   resolution = res_9km,
#   crs = crs(r_1km)
# )
# 
# r_9km <- resample(r_1km, r_9km_template, method = "bilinear")
# 
# plot(r_9km)
# 
# writeRaster(r_9km, "SMAP-HB_9km_daily_mean_20150331.tif", overwrite = TRUE)