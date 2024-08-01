import qupath.lib.images.servers.TransformedServerBuilder

// Get the first selected annotation object
def annotations = getAnnotationObjects()
if (annotations.isEmpty()) {
    print 'No annotations found!'
    return
}
def annotation = annotations[0]

// Define output path
def outputPath = 'D:/Chang_files/workspace/Qupath_proj/hdk_codex/selected_HE.tif'
def outputPath_hema = 'D:/Chang_files/workspace/Qupath_proj/hdk_codex/selected_hema.tif'

// Get current image data and server
def imageData = getCurrentImageData()
def server = new TransformedServerBuilder(imageData.getServer())
    .deconvolveStains(imageData.getColorDeconvolutionStains())
    .build()
    
// Define the region to export
def region = RegionRequest.createInstance(server.getPath(), 1, annotation.getROI())


// Extract Hematoxylin channel (channel 1)
def hemaServer = new TransformedServerBuilder(server)
    .extractChannels(0) // Channel 0 corresponds to Hematoxylin
    .build()

// Save the Hematoxylin channel image
ImageWriterTools.writeImageRegion(hemaServer, region, outputPath_hema)

print 'Export completed: ' + outputPath_hema

def server2 = imageData.getServer()
writeImageRegion(server2, region, outputPath)

print 'Export completed: ' + outputPath
