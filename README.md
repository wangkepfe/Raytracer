# Raytracer
## Geometry model
- [x] ray
- [x] implicit sphere
- [x] implicit aabb
- [x] triangle mesh

## Geometry - Intersection
- [x] ray-sphere
- [x] ray-aabb
- [x] ray-triangle
## Geometry - Smooth surface
## Geometry - Tessellation
## Accelerating Ray Tracing
- [ ] BVH
- [ ] KD-Trees
## Global Illumination
- [x] Monte Carlo Integration
- [x] Path tracing
- [x] Russian Roulette
- [x] Importance sampling
## Surface model
- [x] Diffuse
- [x] Specular
- [x] Mirror
- [x] Transparent
- [ ] Image texture and UV mapping
## Surface advanced model
- [ ] Subsurface Scattering / BSSRDF / Translucent 
- [ ] Anisotropic BRDF
- [ ] Macrofacet BRDF
- [ ] Non-statistical / detailed / glinty BRDF
- [ ] Hair / fur / BCSDF
- [ ] Participating media
- [ ] Cloth
- [ ] Granular
- [ ] Procedural
## Camera model
- [ ] Ideal thin lenses
- [ ] Depth of Field / Bokeh
- [ ] Compound lenses
## Rigging and Animation
## Simulation
## Photon Mapping
## Scene building
- Scene Attributes
	- spheres
	- aabbs
	- triangle meshes
		- vertices
		- faces
		- structure of array (conversion between array of structure)
	- materials
	- geometries
- Geometry
	- type: sphere, aabb, mesh
	- geometry index
	- material index
- Material
	- surface type: diffuse, mirror, transparent
	- emission color
	- surface color

