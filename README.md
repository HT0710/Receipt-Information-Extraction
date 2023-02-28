# Extract Bill Information

### 1. Background removal

- Using **rembg** a library to remove images background by ***danielgatis***
- Source: [rembg](https://github.com/danielgatis/rembg)

### 2. Direction correction

- Using **CRAFT** text detector to get the text direction
- Rotate all horizontal bill to vertical (90 rotate)
- Roate all upside down bill to the correct direction (180 rotate)
