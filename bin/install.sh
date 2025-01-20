chmod +x ./bin/activate.sh
chmod +x ./bin/deactivate.sh
chmod +x ./bin/uninstall.sh
(python3.9 -m venv tmp/venv && source bin/activate.sh && pip install -r requirements.txt)