// Get the first selected annotation object
def annotations = getAnnotationObjects()
if (annotations.isEmpty()) {
    print 'No annotations found!'
    return
}
def annotation = annotations[0]

// Define output path
def outputPath = 'D:/Chang_files/workspace/Qupath_proj/hdk_codex/selected_mihc.tif'

// Get current image data and server
def imageData = getCurrentImageData()
def server = imageData.getServer()

// Define the region to export
def region = RegionRequest.createInstance(server.getPath(), 1, annotation.getROI())

// Export the image region
writeImageRegion(server, region, outputPath)

print 'Export completed: ' + outputPath
