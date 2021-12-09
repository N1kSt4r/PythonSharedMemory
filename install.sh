pip3 install setuptools build
pip3 uninstall shared_store
python3 -m build -w -n
pip3 install dist/*.whl
rm -r dist
rm -r build
rm -r shared_store.egg-info