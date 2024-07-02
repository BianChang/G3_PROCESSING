selectAnnotations()
var pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
def name = getProjectEntry().getImageName() + '.geojson'
print(name)
path = buildFilePath(PROJECT_BASE_DIR, "annotations")
mkdirs(path)
file_name = buildFilePath(path, name)
print(file_name)
exportSelectedObjectsToGeoJson(file_name, "EXCLUDE_MEASUREMENTS", "PRETTY_JSON", "FEATURE_COLLECTION")
println 'Done!'