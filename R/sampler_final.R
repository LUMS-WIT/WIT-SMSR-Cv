library(terra)
library(stringr)
library(fs)

# Define reference top-left coordinates
top_left_coords <- data.frame(
  id = 0:4,
  lon = c(-122.25, -121.876, -120.900, -120.15, -119.551),
  lat = c(39.558, 38.659, 37.534, 36.709, 35.809)
)

# Define input directories
in_root_1km <- "data/SMAP-HB_1km_CV_daily"
in_root_3km <- "data/SMAP-HB_3km_CV_daily"
in_root_9km <- "data/SMAP-HB_9km_CV_daily"

# Output directory
out_root <- "data/training"
dir_create(file.path(out_root, "1km"))
dir_create(file.path(out_root, "3km"))
dir_create(file.path(out_root, "9km"))

# List all 1km files and assume corresponding 3km and 9km files exist
tif_files_1km <- list.files(in_root_1km, pattern = "\\.tif$", recursive = TRUE, full.names = TRUE)

# Utility: get corresponding path in other resolutions
get_corresponding_file <- function(file_1km, from_root, to_root, res_tag_from, res_tag_to) {
  rel_path <- path_rel(file_1km, start = from_root)
  rel_path_new <- str_replace(rel_path, res_tag_from, res_tag_to)
  file.path(to_root, rel_path_new)
}

# Extraction utility
extract_block_by_indices <- function(r, lon, lat, width_px, height_px) {
  cell <- cellFromXY(r, cbind(lon, lat))
  rc <- rowColFromCell(r, cell)
  
  start_row <- rc[1]
  start_col <- rc[2]
  
  end_row <- start_row + height_px - 1
  end_col <- start_col + width_px - 1
  
  top_left <- xyFromCell(r, cellFromRowCol(r, start_row, start_col))
  bottom_right <- xyFromCell(r, cellFromRowCol(r, end_row, end_col))
  
  res_x <- res(r)[1]
  res_y <- res(r)[2]
  
  xmin <- top_left[1] - res_x / 2
  xmax <- bottom_right[1] + res_x / 2
  ymax <- top_left[2] + res_y / 2
  ymin <- bottom_right[2] - res_y / 2
  
  e <- ext(xmin, xmax, ymin, ymax)
  crop(r, e)
}

# Main loop
for (file_1km in tif_files_1km) {
  file_3km <- get_corresponding_file(file_1km, in_root_1km, in_root_3km, "1km", "3km")
  file_9km <- get_corresponding_file(file_1km, in_root_1km, in_root_9km, "1km", "9km")
  
  if (!file.exists(file_3km) || !file.exists(file_9km)) {
    message("Skipping due to missing corresponding files: ", file_1km)
    next
  }
  
  r_1km <- rast(file_1km)
  r_3km <- rast(file_3km)
  r_9km <- rast(file_9km)
  
  date_tag <- str_extract(basename(file_1km), "\\d{8}")  # e.g., 20150401
  
  for (i in 1:nrow(top_left_coords)) {
    lon <- top_left_coords$lon[i]
    lat <- top_left_coords$lat[i]
    id <- top_left_coords$id[i]
    
    block_1km <- extract_block_by_indices(r_1km, lon, lat, 72, 72)
    block_3km <- extract_block_by_indices(r_3km, lon, lat, 24, 24)
    block_9km <- extract_block_by_indices(r_9km, lon, lat, 8, 8)
    
    # Skip if any NA present
    if (
      # any(is.na(values(block_1km))) ||
      # any(is.na(values(block_3km))) ||
      any(is.na(values(block_9km)))
    ) {
      message("Skipped: ", date_tag, " id=", id, " — contains NA")
      next
    }
    
    # Construct output filenames (overwrite original folder with modified filename)
    # Save to CV/training/{res}/filename_id.tif
    base_name_1km <- str_replace(basename(file_1km), "\\.tif$", paste0("_", id, ".tif"))
    base_name_3km <- str_replace(basename(file_3km), "\\.tif$", paste0("_", id, ".tif"))
    base_name_9km <- str_replace(basename(file_9km), "\\.tif$", paste0("_", id, ".tif"))
    
    writeRaster(block_1km, file.path(out_root, "1km", base_name_1km), overwrite = TRUE)
    writeRaster(block_3km, file.path(out_root, "3km", base_name_3km), overwrite = TRUE)
    writeRaster(block_9km, file.path(out_root, "9km", base_name_9km), overwrite = TRUE)
    
    message("Saved: ", date_tag, " id=", id)
  }
}  
