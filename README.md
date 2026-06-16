# Ancient Temple Puzzle Room

A 3D puzzle-exploration game built with **OpenGL + FreeGLUT**, developed as a
Computer Graphics group project (Code::Blocks, single `.cbp` project).

## Concept

You wake up in an ancient stone temple. A light beam emitter sits on the west
wall — guide the beam through three rotating mirror pillars (each must be
turned to a specific angle) to unlock a stone wheel altar. Turning the wheel
opens the great door to a treasure room. Avoid spike traps, swinging
crushers, and arrow turrets along the way — you have 3 lives.

## Controls

| Key | Action |
|---|---|
| `W A S D` / Arrow keys | Move |
| Mouse | Look around |
| `TAB` | Cycle selected pillar |
| `Q` / `E` | Rotate nearby pillar (or crank the wheel) |
| `E` (near chest, door open) | Open the treasure chest |
| `H` | Toggle help overlay |
| `R` | Reset the temple |
| `Esc` | Quit |

## Build instructions (Windows + Code::Blocks)

1. Install [Code::Blocks](https://www.codeblocks.org/) with the **MinGW**
   compiler bundled (the default "codeblocks-XX.XXmingw-setup.exe" installer).
2. Install **FreeGLUT** for MinGW:
   - Download a prebuilt FreeGLUT MinGW package (headers + `libfreeglut.a` /
     `freeglut.dll`).
   - Copy `freeglut.h`, `glut.h` (if present) into
     `<MinGW>/include/GL/`.
   - Copy `libfreeglut.a` (and `libfreeglut_static.a` if you have it) into
     `<MinGW>/lib/`.
   - Copy `freeglut.dll` into your project's output folder (or
     `C:\Windows\System32`) so the `.exe` can find it at runtime.
3. Open `TempleRoom.cbp` in Code::Blocks.
4. Build → Build and Run (`Ctrl+F9` / `F9`). The project already has the
   linker flags `-lfreeglut -lopengl32 -lglu32` configured.

## Build instructions (Linux)

```bash
sudo apt-get install freeglut3-dev
g++ -std=c++17 -O2 main.cpp -o TempleRoom -lglut -lGL -lGLU
./TempleRoom
```

(On Linux, `-lglut` is used instead of `-lfreeglut`.)

## Project structure

```
.
├── main.cpp          # All game source code (single-file project)
├── TempleRoom.cbp    # Code::Blocks project file
├── .gitignore
└── README.md
```

## Graphics techniques used

- Phong illumination (8 OpenGL lights: torches, beam spotlight, victory light)
- Translation / rotation / scaling via the matrix stack
- View transformation (mouse-look camera with `gluLookAt`)
- Perspective projection (`gluPerspective`)
- DDA-style incremental floor tiling
- AABB collision/clipping for the player and traps
- 2D HUD overlay via `gluOrtho2D`
- State-machine animation (spike toggle, crusher phase timer)
- Smooth interpolation (pillar rotation lerp, chest lid lerp)
- Particle system (position, velocity, gravity, fade)

## License

MIT (or your course's preferred license — update as needed).
