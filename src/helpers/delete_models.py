import hopsworks

project = hopsworks.login()

mr = project.get_model_registry()

model_names =["tensorflow_nn", "lightgbm", "xgboost", "random_forest", "elasticnet"]

for model_name in model_names:
    registered_models = mr.get_models(name=model_name)
    model_name_to_delete = model_name
    if not registered_models:
        print(f"No registered models found with name: {model_name}")
    else:
        print(f"Found {len(registered_models)} registered model(s) with name: {model_name}")
    
    for model in registered_models:
        if model.name == model_name_to_delete:
            print(f"Deleting model version: {model.version} for model: {model_name_to_delete}")
            try:
                model.delete()
                print(f"Successfully deleted model version: {model.version}")
            except Exception as e:
                print(f"Failed to delete model version {model.version}: {e}")



