from eppa_viz.webapp.app_factory import create_app

app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
