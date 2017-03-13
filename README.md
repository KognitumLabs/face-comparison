# Setup and download models

```
sh setup.sh
```

# Install

```
docker build -t face-comparison .
docker run -d -p 5000:5000 face-comparison
```

# Run in dev

```
 python app.py -p 8080
``` 