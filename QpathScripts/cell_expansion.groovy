import qupath.ext.stardist.StarDist2D

// Specify the model file (you will need to change this!)
var pathModel = 'D:/Chang_files/workspace/Qupath_proj/dsb2018_heavy_augment.pb'

var stardist = StarDist2D.builder(pathModel)
        .threshold(0.5)              // Probability (detection) threshold
        .channels('DAPI 100% - 345/455')            // Select detection channel
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(0.5)              // Resolution for detection
        .cellExpansion(5.0)          // Approximate cells based upon nucleus expansion
        .cellConstrainScale(1.5)     // Constrain cell expansion using nucleus size
        .measureShape()              // Add shape measurements
        .measureIntensity()          // Add cell measurements (in all compartments)
        .includeProbability(true)    // Add probability as a measurement (enables later filtering)
        .build()

// Run detection for the selected objects
var imageData = getCurrentImageData()
selectAnnotations()
var pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
stardist.detectObjects(imageData, pathObjects)
println 'Done!'