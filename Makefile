PLUGIN_NAME = roe_deer_corridor_demo
REPO_DIR    = ../QGIS-Plugin-Repository

.PHONY: publish zip clean

publish: zip
	cp $(PLUGIN_NAME).zip $(REPO_DIR)/plugins/
	python $(REPO_DIR)/generate_xml.py --base-url https://zhaw-qgis-plugins.github.io/Plugins-Repository/plugins

zip: clean
	zip -r $(PLUGIN_NAME).zip $(PLUGIN_NAME)/

clean:
	rm -f $(PLUGIN_NAME).zip
