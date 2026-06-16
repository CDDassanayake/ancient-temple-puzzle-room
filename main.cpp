// Ancient Temple Puzzle Room
// OpenGL + FreeGLUT   |   Code::Blocks project: TempleRoom.cbp
// Linker flags: -lfreeglut -lopengl32 -lglu32

// Graphics concepts applied across the project:
//   Phong illumination  (ambient, diffuse, specular on 8 OpenGL lights)
//   Translation, rotation, scaling  (every glPushMatrix/glTranslate/glRotate/glScale call)
//   View transformation  (gluLookAt updated each frame from Euler yaw+pitch angles)
//   Perspective projection  (gluPerspective 65 degree FOV)
//   DDA-style incremental stepping  (floor tile loop, beam step-march)
//   AABB clipping  (camera clamped inside room walls, object collision push)
//   State-machine animation  (spike bool toggle, crusher continuous phase timer)
//   Smooth interpolation  (pillar rotation lerp, chest lid lerp)
//   Particle system  (position + velocity + gravity + fade per particle)

#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>

struct V3 {
    float x,y,z;
    V3(float x=0,float y=0,float z=0):x(x),y(y),z(z){}
    V3 operator+(const V3& o)const{return{x+o.x,y+o.y,z+o.z};}
    V3 operator-(const V3& o)const{return{x-o.x,y-o.y,z-o.z};}
    V3 operator*(float t)   const{return{x*t,y*t,z*t};}
    V3& operator+=(const V3& o){x+=o.x;y+=o.y;z+=o.z;return*this;}
    V3& operator-=(const V3& o){x-=o.x;y-=o.y;z-=o.z;return*this;}
    float dot(const V3& o)const{return x*o.x+y*o.y+z*o.z;}
    float len()const{return sqrtf(x*x+y*y+z*z);}
    V3 norm()const{float l=len();return l>0.001f?V3(x/l,y/l,z/l):*this;}
    V3 cross(const V3& o)const{return{y*o.z-z*o.y,z*o.x-x*o.z,x*o.y-y*o.x};}
};
static inline float RAD(float d){return d*0.01745329f;}
static inline float CLAMP(float v,float lo,float hi){return v<lo?lo:(v>hi?hi:v);}
static inline float LERP(float a,float b,float t){return a+(b-a)*t;}

const int SCR_W = 1280, SCR_H = 720;
const float ROOM_W = 14.f;
const float ROOM_D = 34.f;
const float ROOM_H =  6.5f;
const int   MAX_LIVES = 3;

const float PILLAR_INTERACT_RADIUS = 2.2f;
const float SOLUTION[3] = { 90.f, 135.f, 315.f };
const float SNAP_STEP   = 45.f;
const float SOLVE_TOL   = 8.f;

struct Particle {
    V3    pos, vel;
    float r,g,b,a, life, maxLife, size;
};

struct TorchLight {
    V3    pos;
    float flicker;
};

struct Pillar {
    V3    pos;
    float rotY;
    float targetRotY;
    bool  crystalLit;
    bool  showGuide;
    int   solveStep;
};

struct Arrow {
    V3    pos, dir;
    float life, speed;
    bool  active;
};

enum Phase { PHASE_EXPLORE, PHASE_DEAD, PHASE_WIN };

static Phase       g_phase       = PHASE_EXPLORE;
static int         g_lives       = MAX_LIVES;
static float       g_time        = 0.f;
static float       g_dt          = 0.016f;
static bool        g_puzzleDone  = false;
static float       g_solveTimer  = 0.f;
static float       g_deadTimer   = 0.f;
static std::string g_msg         = "";
static float       g_msgTimer    = 0.f;
static bool        g_showHelp    = false;
static int         g_selPillar   = 0;

static V3    g_camPos(0, 1.8f, 9.f);
static V3    g_camFront(0, 0, -1);
static V3    g_camUp(0, 1, 0);
static float g_yaw   = -90.f;
static float g_pitch =  -5.f;
static int   g_mLastX = SCR_W/2, g_mLastY = SCR_H/2;
static bool  g_firstMouse = true;

static std::vector<Pillar>     g_pillars;
static std::vector<TorchLight> g_torches;
static std::vector<Particle>   g_particles;
static std::vector<Arrow>      g_arrows;

static float g_spikeOffset = 0.f;
static bool  g_spikesUp    = true;
static float g_arrowTimer  = 2.f;
static float g_chestLid    = 0.f;
static bool  g_chestOpening= false;
static float g_wheelRot    = 0.f;
static float g_doorY       = 0.f;
static float g_beamGlow    = 0.f;
static bool  g_dmgCooldown = false;
static float g_dmgTimer    = 0.f;

static V3    g_spawnPos(0, 1.8f, 9.f);
static int   g_nearPillar  = -1;
static float g_nearDist    = 999.f;
static float g_walkPhase   = 0.f;

void showMessage(const std::string& s, float dur=3.f);
void respawn();
void checkTrapCollisions();
bool checkBeamSolved();
void spawnParticles(V3 origin,int n,float r,float g,float b,float spread=1.f);
void updateNearestPillar();

// MEMBER1  All scene geometry — room, pillars, traps, chest, wheel altar
// Techniques: output primitives (box, cylinder, cone, sphere), translation,
// rotation, scaling, DDA-style incremental floor tiling, pit region check

static void drawBox(float w, float h, float d){
    float hw=w*.5f, hh=h*.5f, hd=d*.5f;
    glBegin(GL_QUADS);
    glNormal3f(0,0,1);
    glVertex3f(-hw,-hh,hd);glVertex3f(hw,-hh,hd);
    glVertex3f(hw,hh,hd);glVertex3f(-hw,hh,hd);
    glNormal3f(0,0,-1);
    glVertex3f(hw,-hh,-hd);glVertex3f(-hw,-hh,-hd);
    glVertex3f(-hw,hh,-hd);glVertex3f(hw,hh,-hd);
    glNormal3f(-1,0,0);
    glVertex3f(-hw,-hh,-hd);glVertex3f(-hw,-hh,hd);
    glVertex3f(-hw,hh,hd);glVertex3f(-hw,hh,-hd);
    glNormal3f(1,0,0);
    glVertex3f(hw,-hh,hd);glVertex3f(hw,-hh,-hd);
    glVertex3f(hw,hh,-hd);glVertex3f(hw,hh,hd);
    glNormal3f(0,1,0);
    glVertex3f(-hw,hh,hd);glVertex3f(hw,hh,hd);
    glVertex3f(hw,hh,-hd);glVertex3f(-hw,hh,-hd);
    glNormal3f(0,-1,0);
    glVertex3f(-hw,-hh,-hd);glVertex3f(hw,-hh,-hd);
    glVertex3f(hw,-hh,hd);glVertex3f(-hw,-hh,hd);
    glEnd();
}

static void drawCylinder(float r, float h, int sl){
    float s=2.f*3.14159f/sl;
    glBegin(GL_QUAD_STRIP);
    for(int i=0;i<=sl;i++){
        float a=i*s;
        glNormal3f(cosf(a),0,sinf(a));
        glVertex3f(cosf(a)*r,0,sinf(a)*r);
        glVertex3f(cosf(a)*r,h,sinf(a)*r);
    }
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0,1,0);glVertex3f(0,h,0);
    for(int i=0;i<=sl;i++){float a=i*s;glVertex3f(cosf(a)*r,h,sinf(a)*r);}
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0,-1,0);glVertex3f(0,0,0);
    for(int i=sl;i>=0;i--){float a=i*s;glVertex3f(cosf(a)*r,0,sinf(a)*r);}
    glEnd();
}

static void drawSphere(float r, int sl, int st){
    for(int i=0;i<st;i++){
        float la=3.14159f*(-0.5f+(float)i/st);
        float lb=3.14159f*(-0.5f+(float)(i+1)/st);
        glBegin(GL_QUAD_STRIP);
        for(int j=0;j<=sl;j++){
            float lo=2*3.14159f*(float)j/sl;
            glNormal3f(cosf(la)*cosf(lo),sinf(la),cosf(la)*sinf(lo));
            glVertex3f(cosf(la)*cosf(lo)*r,sinf(la)*r,cosf(la)*sinf(lo)*r);
            glNormal3f(cosf(lb)*cosf(lo),sinf(lb),cosf(lb)*sinf(lo));
            glVertex3f(cosf(lb)*cosf(lo)*r,sinf(lb)*r,cosf(lb)*sinf(lo)*r);
        }
        glEnd();
    }
}

static void drawCone(float baseR, float tipH, int sl){
    float s=2.f*3.14159f/sl;
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0,1,0); glVertex3f(0,tipH,0);
    for(int i=0;i<=sl;i++){
        float a=i*s;
        glNormal3f(cosf(a),0.5f,sinf(a));
        glVertex3f(cosf(a)*baseR,0,sinf(a)*baseR);
    }
    glEnd();
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0,-1,0); glVertex3f(0,0,0);
    for(int i=sl;i>=0;i--){float a=i*s;glVertex3f(cosf(a)*baseR,0,sinf(a)*baseR);}
    glEnd();
}

// Returns true if a world XZ position sits over one of the floor pits.
// Pits flank each spike row so the player can fall off if not careful.
static bool isFloorHole(float x, float z){
    if(z > 1.3f && z < 3.7f && (x < -3.0f || x > 3.0f)) return true;
    if(z > -5.7f && z < -3.3f && (x < -1.2f || x > 3.8f)) return true;
    return false;
}

static void drawRoom(){
    float hw=ROOM_W*.5f, hd=ROOM_D*.5f, H=ROOM_H;

    // Floor — DDA-style: fx advances by one tile width each step, no per-tile division
    float tile=1.4f;
    int tx=(int)(ROOM_W/tile)+2, tz=(int)(ROOM_D/tile)+2;
    for(int i=0;i<tx;i++){
        for(int j=0;j<tz;j++){
            float fx=-hw+i*tile, fz=-hd+j*tile;
            float cx=fx+tile*0.5f, cz=fz+tile*0.5f;
            if(isFloorHole(cx, cz)){
                glColor3f(.02f,.01f,.01f);
                glBegin(GL_QUADS);
                glNormal3f(0,1,0);
                glVertex3f(fx,      -0.3f, fz);
                glVertex3f(fx+tile, -0.3f, fz);
                glVertex3f(fx+tile, -0.3f, fz+tile);
                glVertex3f(fx,      -0.3f, fz+tile);
                glEnd();
                glColor3f(.10f,.07f,.04f);
                glBegin(GL_QUADS); glNormal3f(0,0,1);
                glVertex3f(fx,0,fz); glVertex3f(fx+tile,0,fz);
                glVertex3f(fx+tile,-0.3f,fz); glVertex3f(fx,-0.3f,fz);
                glEnd();
                glBegin(GL_QUADS); glNormal3f(0,0,-1);
                glVertex3f(fx,0,fz+tile); glVertex3f(fx+tile,0,fz+tile);
                glVertex3f(fx+tile,-0.3f,fz+tile); glVertex3f(fx,-0.3f,fz+tile);
                glEnd();
                continue;
            }
            float base = ((i+j)%2==0) ? 0.28f : 0.38f;
            float shimmer = 0.02f * sinf(i*1.3f + j*0.9f);
            glColor3f(base+shimmer+0.04f, base+shimmer, base+shimmer*0.5f);
            glBegin(GL_QUADS);
            glNormal3f(0,1,0);
            glVertex3f(fx,      0, fz);
            glVertex3f(fx+tile, 0, fz);
            glVertex3f(fx+tile, 0, fz+tile);
            glVertex3f(fx,      0, fz+tile);
            glEnd();
        }
    }
    glColor3f(.20f,.15f,.10f);
    glPushMatrix(); glTranslatef(0,.01f,0);
    for(float tz2:{-hd,hd-0.1f}){
        glPushMatrix(); glTranslatef(0,0,tz2); drawBox(ROOM_W,.06f,.1f); glPopMatrix();
    }
    for(float tx2:{-hw,hw-0.1f}){
        glPushMatrix(); glTranslatef(tx2,0,0); drawBox(.1f,.06f,ROOM_D); glPopMatrix();
    }
    glPopMatrix();

    glColor3f(.18f,.13f,.09f);
    glBegin(GL_QUADS);
    glNormal3f(0,-1,0);
    glVertex3f(-hw,H,-hd); glVertex3f(hw,H,-hd);
    glVertex3f(hw,H,hd);   glVertex3f(-hw,H,hd);
    glEnd();
    glColor3f(.14f,.10f,.07f);
    for(int b=0;b<6;b++){
        glPushMatrix();
        glTranslatef(0,H-.06f,-hd+1.8f+b*(ROOM_D-2.f)/5.f);
        drawBox(ROOM_W,.3f,.4f);
        glPopMatrix();
    }

    glColor3f(.26f,.20f,.13f);
    glBegin(GL_QUADS);
    glNormal3f(0,0,1);
    glVertex3f(-hw,0,-hd); glVertex3f(hw,0,-hd);
    glVertex3f(hw,H,-hd);  glVertex3f(-hw,H,-hd);
    glEnd();
    glColor3f(.18f,.13f,.08f);
    glBegin(GL_QUADS); glNormal3f(0,0,1);
    glVertex3f(-3.2f,.15f,-hd+.02f); glVertex3f(3.2f,.15f,-hd+.02f);
    glVertex3f(3.2f,4.8f,-hd+.02f);  glVertex3f(-3.2f,4.8f,-hd+.02f);
    glEnd();
    glColor3f(.34f,.25f,.14f);
    for(int k=0;k<7;k++){
        float yy=.5f+k*.6f;
        glBegin(GL_LINES);
        glVertex3f(-2.8f,yy,-hd+.03f); glVertex3f(2.8f,yy,-hd+.03f);
        glEnd();
    }
    glColor3f(.30f,.22f,.12f);
    for(int k=-2;k<=2;k++){
        float xx=k*.9f;
        glBegin(GL_LINES);
        glVertex3f(xx,.4f,-hd+.03f); glVertex3f(xx,4.5f,-hd+.03f);
        glEnd();
    }

    glColor3f(.26f,.20f,.13f);
    glBegin(GL_QUADS); glNormal3f(0,0,-1);
    glVertex3f(-hw,0,hd);   glVertex3f(-1.4f,0,hd);
    glVertex3f(-1.4f,H,hd); glVertex3f(-hw,H,hd);
    glEnd();
    glBegin(GL_QUADS); glNormal3f(0,0,-1);
    glVertex3f(1.4f,0,hd);  glVertex3f(hw,0,hd);
    glVertex3f(hw,H,hd);    glVertex3f(1.4f,H,hd);
    glEnd();
    glBegin(GL_QUADS); glNormal3f(0,0,-1);
    glVertex3f(-1.4f,3.4f,hd); glVertex3f(1.4f,3.4f,hd);
    glVertex3f(1.4f,H,hd);     glVertex3f(-1.4f,H,hd);
    glEnd();

    glColor3f(.24f,.18f,.12f);
    glBegin(GL_QUADS); glNormal3f(1,0,0);
    glVertex3f(-hw,0,-hd); glVertex3f(-hw,0,hd);
    glVertex3f(-hw,H,hd);  glVertex3f(-hw,H,-hd);
    glEnd();
    glBegin(GL_QUADS); glNormal3f(-1,0,0);
    glVertex3f(hw,0,hd);  glVertex3f(hw,0,-hd);
    glVertex3f(hw,H,-hd); glVertex3f(hw,H,hd);
    glEnd();

    glColor3f(.30f,.23f,.15f);
    float pilZ[]={-8.f,-4.f,0.f,4.f,8.f};
    for(float pz:pilZ){
        glPushMatrix(); glTranslatef(-hw+.12f,0,pz);
        drawCylinder(.22f,H,12); glPopMatrix();
        glPushMatrix(); glTranslatef(hw-.12f,0,pz);
        drawCylinder(.22f,H,12); glPopMatrix();
        glPushMatrix(); glTranslatef(-hw+.12f,H-.3f,pz);
        glTranslatef(-.2f,0,-.2f); drawBox(.4f,.3f,.4f); glPopMatrix();
        glPushMatrix(); glTranslatef(hw-.12f,H-.3f,pz);
        glTranslatef(-.2f,0,-.2f); drawBox(.4f,.3f,.4f); glPopMatrix();
    }

    // Wall sconces — the bracket and bowl that hold each torch
    float torchZ[]={-6.f,-2.f,2.f,6.f};
    for(float tz3:torchZ){
        for(int s:{-1,1}){
            glPushMatrix();
            glTranslatef(s*(hw-.06f), 3.2f, tz3);
            glColor3f(.38f,.28f,.14f);
            drawBox(.20f,.10f,.42f);
            glTranslatef(0,.15f,.16f);
            glColor3f(.22f,.16f,.08f);
            drawCylinder(.072f,.22f,8);
            glPopMatrix();
        }
    }

    glColor3f(.32f,.25f,.17f);
    glPushMatrix(); glTranslatef(0,3.3f,hd+.01f); drawBox(2.8f,.28f,.28f); glPopMatrix();
    glPushMatrix(); glTranslatef(-1.4f,1.65f,hd+.01f); drawBox(.26f,3.3f,.28f); glPopMatrix();
    glPushMatrix(); glTranslatef( 1.4f,1.65f,hd+.01f); drawBox(.26f,3.3f,.28f); glPopMatrix();

    // Beam emitter mounted on the left wall
    glColor3f(.45f,.35f,.18f);
    glPushMatrix();
    glTranslatef(-hw+.12f, 3.1f, 0.f);
    drawBox(.25f,.25f,.38f);
    glColor3f(.2f,.6f,1.f);
    glPushMatrix(); glTranslatef(.14f,0,0); drawCylinder(.07f,.06f,12); glPopMatrix();
    glPopMatrix();

    // Inner dividing wall with sliding door (z = -10)
    glColor3f(.22f, .18f, .14f);
    glPushMatrix(); glTranslatef(-4.25f, ROOM_H/2, -10.0f); drawBox(5.5f, ROOM_H, 0.4f); glPopMatrix();
    glPushMatrix(); glTranslatef(4.25f, ROOM_H/2, -10.0f); drawBox(5.5f, ROOM_H, 0.4f); glPopMatrix();
    glPushMatrix(); glTranslatef(0, ROOM_H - 1.0f, -10.0f); drawBox(3.0f, 2.0f, 0.4f); glPopMatrix();

    glColor3f(.30f, .26f, .22f);
    glPushMatrix();
    glTranslatef(0, (ROOM_H/2 - 1.0f) + g_doorY, -10.0f);
    drawBox(3.2f, ROOM_H - 2.0f, 0.3f);
    glColor3f(.20f, .16f, .12f);
    glTranslatef(0,0,0.16f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-1.4f, -2.0f, 0); glVertex3f(1.4f, -2.0f, 0);
    glVertex3f(1.4f, 2.0f, 0);   glVertex3f(-1.4f, 2.0f, 0);
    glEnd();
    glTranslatef(0, 0, 0.02f);
    glBegin(GL_LINE_LOOP);
    for(int i=0;i<16;i++) {
        float a=i*3.14159f*2.0f/16.0f;
        glVertex3f(cosf(a)*0.6f, sinf(a)*0.6f, 0);
    }
    glEnd();
    glPopMatrix();
}

static void drawPillars(){
    for(int i=0;i<(int)g_pillars.size();i++){
        const Pillar& p=g_pillars[i];
        glPushMatrix();
        glTranslatef(p.pos.x, p.pos.y, p.pos.z);
        glRotatef(p.rotY, 0,1,0);

        glColor3f(.34f,.26f,.18f);
        glPushMatrix(); glTranslatef(0,.15f,0); drawBox(.95f,.3f,.95f); glPopMatrix();

        bool sel    = (i==g_selPillar);
        bool isNear = (i==g_nearPillar);
        if(p.crystalLit)       glColor3f(.60f,.55f,.28f);
        else if(sel && isNear) glColor3f(.52f,.45f,.30f);
        else if(isNear)        glColor3f(.44f,.38f,.24f);
        else                   glColor3f(.36f,.28f,.20f);

        glPushMatrix(); glTranslatef(0,.3f,0); drawCylinder(.30f,2.5f,18); glPopMatrix();

        glColor3f(.28f,.21f,.14f);
        for(float ry:{.85f,1.5f,2.15f}){
            glPushMatrix(); glTranslatef(0,.3f+ry,0); drawCylinder(.33f,.09f,16); glPopMatrix();
        }

        glColor3f(.32f,.25f,.17f);
        glPushMatrix(); glTranslatef(0,2.82f,0); drawBox(1.0f,.32f,1.0f); glPopMatrix();

        glColor3f(.26f,.19f,.12f);
        glPushMatrix(); glTranslatef(0, 2.98f, 0); drawCylinder(.34f, .10f, 16); glPopMatrix();
        glColor3f(.14f,.10f,.07f);
        glPushMatrix(); glTranslatef(0, 2.99f, 0); drawCylinder(.22f, .08f, 16); glPopMatrix();

        if(p.crystalLit)      glColor3f(1.f,.88f,.15f);
        else if(p.showGuide)  glColor3f(.20f,.85f,.45f);
        else                  glColor3f(.38f,.48f,.72f);
        glPushMatrix();
        glTranslatef(0, 3.12f, 0);
        glRotatef(g_time*30.f, 0,1,0);
        glRotatef(45,1,0,0);
        drawBox(.22f,.22f,.22f);
        glPopMatrix();

        // Small cone shows which direction the mirror face points
        glColor3f(.9f,.7f,.1f);
        glPushMatrix();
        glTranslatef(0.f, 1.6f, 0.20f);
        glRotatef(90,1,0,0);
        drawCone(.08f,.22f,8);
        glPopMatrix();

        if(sel && isNear){
            glColor3f(1.f,.88f,.1f);
            glPushMatrix(); glTranslatef(0,.04f,0);
            glBegin(GL_LINE_LOOP);
            for(int k=0;k<24;k++){
                float a=k*2.f*3.14159f/24;
                glVertex3f(cosf(a)*.65f,.01f,sinf(a)*.65f);
            }
            glEnd();
            glPopMatrix();
        } else if(sel && !isNear){
            glColor3f(.5f,.4f,.1f);
            glPushMatrix(); glTranslatef(0,.04f,0);
            glBegin(GL_LINE_LOOP);
            for(int k=0;k<24;k++){
                float a=k*2.f*3.14159f/24;
                glVertex3f(cosf(a)*.65f,.01f,sinf(a)*.65f);
            }
            glEnd();
            glPopMatrix();
        }

        glPopMatrix();
    }
}

static void drawTreasureChest(){
    glPushMatrix();
    glTranslatef(0,.01f,-15.0f);

    glColor3f(.42f,.27f,.11f);
    glPushMatrix(); glTranslatef(0,.32f,0); drawBox(1.2f,.62f,.82f); glPopMatrix();

    glColor3f(.68f,.54f,.20f);
    for(float bx:{-.42f,0.f,.42f}){
        glPushMatrix(); glTranslatef(bx,.32f,0); drawBox(.07f,.64f,.84f); glPopMatrix();
    }
    glPushMatrix(); glTranslatef(0,.42f,0); drawBox(1.22f,.07f,.84f); glPopMatrix();

    if(g_puzzleDone) glColor3f(1.f,1.f,.3f);
    else             glColor3f(.9f,.8f,.1f);
    glPushMatrix(); glTranslatef(0,.54f,.43f); drawBox(.18f,.18f,.07f); glPopMatrix();
    glPushMatrix(); glTranslatef(0,.44f,.44f); drawBox(.10f,.10f,.05f); glPopMatrix();

    // Gold coins inside the chest — only drawn once the lid is far enough open
    if(g_chestLid > 30.f){
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        float shimmer = 0.75f + 0.25f * sinf(g_time * 5.f);
        for(int c=0;c<12;c++){
            float ang = c * 30.f;
            float rad = 0.05f + (c % 3) * 0.12f;
            glColor4f(1.f * shimmer, 0.80f * shimmer, 0.05f, 0.95f);
            glPushMatrix();
            glTranslatef(cosf(RAD(ang))*rad, .64f, sinf(RAD(ang))*rad*.6f);
            drawCylinder(.072f,.045f,10);
            glPopMatrix();
        }
        for(int c=0;c<8;c++){
            float ang = c * 45.f + 15.f;
            float rad = 0.18f + (c % 2) * 0.08f;
            glColor4f(1.f * shimmer, 0.72f * shimmer, 0.02f, 0.90f);
            glPushMatrix();
            glTranslatef(cosf(RAD(ang))*rad, .68f, sinf(RAD(ang))*rad*.55f);
            drawCylinder(.068f,.04f,10);
            glPopMatrix();
        }
        float gemPulse = 0.8f + 0.2f * sinf(g_time * 6.f);
        glColor4f(1.f, 0.92f * gemPulse, 0.1f * gemPulse, 1.f);
        glPushMatrix();
        glTranslatef(0.f, .80f, 0.f);
        glRotatef(g_time * 80.f, 0,1,0);
        glRotatef(45,1,0,0);
        drawBox(.18f,.18f,.18f);
        glPopMatrix();

        for(int ring=0;ring<3;ring++){
            float rr = (.28f + ring*.14f);
            glColor4f(1.f, .82f, .15f, (.5f - ring*.14f) * shimmer);
            glPushMatrix();
            glTranslatef(0,.70f,0);
            glBegin(GL_LINE_LOOP);
            for(int k=0;k<24;k++){
                float a=k*2.f*3.14159f/24;
                glVertex3f(cosf(a)*rr, 0, sinf(a)*rr*.65f);
            }
            glEnd();
            glPopMatrix();
        }

        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }

    glPushMatrix();
    glTranslatef(0,.63f,-.41f);
    glRotatef(-g_chestLid,1,0,0);
    glTranslatef(0,.16f,.41f);
    glColor3f(.48f,.31f,.13f);
    drawBox(1.2f,.32f,.82f);
    glColor3f(.68f,.54f,.20f);
    for(float bx:{-.42f,0.f,.42f}){
        glPushMatrix(); glTranslatef(bx,0,0); drawBox(.07f,.34f,.84f); glPopMatrix();
    }
    glPopMatrix();

    if(g_chestLid > 30.f){
        glColor3f(1.f,.82f,.10f);
        for(int c=0;c<10;c++){
            float ang=c*36.f + 8.f;
            float rad=0.55f+(c%3)*0.18f;
            glPushMatrix();
            glTranslatef(cosf(RAD(ang))*rad, .02f, sinf(RAD(ang))*rad*.7f);
            glRotatef(ang*3.f,0,1,0);
            drawCylinder(.07f,.03f,8);
            glPopMatrix();
        }
    }

    glPopMatrix();
}

static void drawTraps(){
    // Three crusher blocks that drop from the ceiling on chains.
    // Each block has a different phaseOff so they are never all at floor level simultaneously.
    // Crusher 0 is at (-0.8, *, 4.8) — north of the spike row and clear of pillar 1
    // and both pit regions, leaving a visible gap the player can dodge through.
    struct CrusherTrap { float x, z, phaseOff; };
    CrusherTrap crushers[]={ {-0.8f, .8f, 4.8f}, {3.2f, -1.2f, 0.33f}, {0.0f, -5.5f, 0.66f} };
    for (auto& c : crushers) {
        float phase = fmodf(g_time * 0.4f + c.phaseOff, 1.0f);
        float h = ROOM_H;
        if (phase < 0.1f) h = ROOM_H - (ROOM_H * (phase * 10.0f));
        else if (phase < 0.3f) h = 0.0f;
        else h = (phase - 0.3f) / 0.7f * ROOM_H;

        glPushMatrix();
        glTranslatef(c.x, h, c.z);
        glColor3f(.15f,.15f,.15f);
        glPushMatrix(); glTranslatef(0, 1.2f, 0); drawCylinder(0.04f, ROOM_H - (h + 1.2f), 8); glPopMatrix();
        glColor3f(.22f,.18f,.14f);
        glTranslatef(0, 0.6f, 0);
        drawBox(1.2f, 1.2f, 1.2f);
        glColor3f(.4f, .1f, .1f);
        glTranslatef(0, -0.58f, 0);
        drawBox(1.22f, 0.1f, 1.22f);
        glPopMatrix();
    }

    float y1 = LERP(-0.95f, -0.05f, g_spikeOffset);
    float y2 = LERP(-0.05f, -0.95f, g_spikeOffset);

    for(int s=-4;s<=4;s++){
        glPushMatrix();
        glTranslatef(s*.6f, y1, 2.5f);
        glColor3f(.15f, .15f, .15f);
        drawCylinder(.14f, .06f, 16);
        glColor3f(.75f, .78f, .82f);
        glTranslatef(0, .06f, 0);
        drawCone(.11f, .85f, 16);
        glPopMatrix();
    }

    for(int s=-3;s<=3;s++){
        glPushMatrix();
        glTranslatef(s*.6f+1.5f, y2, -4.5f);
        glColor3f(.15f, .15f, .15f);
        drawCylinder(.14f, .06f, 16);
        glColor3f(.75f, .78f, .82f);
        glTranslatef(0, .06f, 0);
        drawCone(.11f, .85f, 16);
        glPopMatrix();
    }

    float hw2=ROOM_W*.5f;
    glColor3f(.12f,.08f,.05f);
    for(float az:{-3.5f,-.5f}){
        glPushMatrix(); glTranslatef(-hw2+.01f,1.3f,az); drawBox(.05f,.14f,.40f); glPopMatrix();
        glPushMatrix(); glTranslatef( hw2-.01f,1.3f,az); drawBox(.05f,.14f,.40f); glPopMatrix();
    }

    for(auto& ar:g_arrows){
        if(!ar.active) continue;
        glPushMatrix();
        glTranslatef(ar.pos.x,ar.pos.y,ar.pos.z);
        float angY=atan2f(ar.dir.x,ar.dir.z)*180.f/3.14159f;
        glRotatef(angY,0,1,0);
        glRotatef(90.f,1,0,0);
        glTranslatef(0, -0.29f, 0);
        glColor3f(.52f,.36f,.14f);
        drawCylinder(.028f,.58f,6);
        glColor3f(.72f,.62f,.32f);
        glPushMatrix(); glTranslatef(0,.58f,0); drawCone(.055f,.20f,8); glPopMatrix();
        glColor3f(.88f,.82f,.72f);
        glPushMatrix(); glTranslatef(.04f,.08f,0); drawBox(.08f,.24f,.02f); glPopMatrix();
        glPushMatrix(); glTranslatef(-.04f,.08f,0); drawBox(.08f,.24f,.02f); glPopMatrix();
        glPopMatrix();
    }
}

static void drawWheelPillar(){
    glPushMatrix();
    glTranslatef(0, 0, -8.5f);
    glRotatef(g_wheelRot, 0, 1, 0);

    glColor3f(.28f,.21f,.14f);
    glPushMatrix(); glTranslatef(0,0,0); drawCylinder(.52f,1.1f,18); glPopMatrix();
    glColor3f(.32f,.24f,.16f);
    glPushMatrix(); glTranslatef(0,1.1f,0); drawCylinder(.62f,.14f,18); glPopMatrix();

    glColor3f(.22f,.18f,.12f);
    for(int i=0; i<4; i++) {
        glPushMatrix();
        glRotatef(i*90.0f, 0, 1, 0);
        glTranslatef(.75f, 1.1f, 0);
        drawBox(.3f, .08f, .08f);
        glPopMatrix();
    }

    if(g_puzzleDone) glColor3f(1.f,.90f,.18f);
    else             glColor3f(.32f,.42f,.68f);
    glPushMatrix();
    glTranslatef(0,1.38f,0);
    glRotatef(g_time*65.f,0,1,0);
    drawSphere(.28f,18,18);
    glPopMatrix();

    glColor3f(.42f,.32f,.18f);
    for(int k=0;k<12;k++){
        float a=k*2.f*3.14159f/12;
        glPushMatrix();
        glTranslatef(cosf(a)*.72f, 1.12f, sinf(a)*.72f);
        glRotatef(a*180.f/3.14159f,0,1,0);
        drawBox(.05f,.08f,.12f);
        glPopMatrix();
    }

    glPopMatrix();
}

// MEMBER2  Rendering effects — beam glow, fixed torch flame, particles, HUD
// The beam uses a step-march loop (same idea as Bresenham: advance by a fixed
// increment and sample at each step rather than computing every pixel from scratch).
// Torch flames are now fixed 3D cones — colour-only flicker, no movement.
// HUD uses gluOrtho2D so everything is drawn in screen pixel coordinates.

static void drawBeamGlow(){
    if(g_pillars.size()<3) return;

    V3 src(-ROOM_W*.5f+.26f, 3.1f, 0.f);
    V3 pts[5];
    pts[0]=src;
    for(int i=0;i<3;i++) pts[i+1]=V3(g_pillars[i].pos.x, g_pillars[i].pos.y+3.12f, g_pillars[i].pos.z);
    pts[4]=V3(0.f, 1.38f, -8.5f);

    int segs=0;
    if(g_pillars[0].crystalLit) segs=1;
    if(segs==1&&g_pillars[1].crystalLit) segs=2;
    if(segs==2&&g_pillars[2].crystalLit) segs=3;
    if(segs==3) segs=4;

    if(segs==0) return;

    float alpha=.45f+.30f*sinf(g_beamGlow*4.2f);

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);

    glLineWidth(12.f);
    glColor4f(.6f,.8f,1.f, alpha*.15f);
    glBegin(GL_LINE_STRIP);
    for(int s=0;s<=segs;s++) glVertex3f(pts[s].x,pts[s].y,pts[s].z);
    glEnd();

    glLineWidth(5.f);
    glColor4f(.8f,.9f,1.f, alpha*.45f);
    glBegin(GL_LINE_STRIP);
    for(int s=0;s<=segs;s++) glVertex3f(pts[s].x,pts[s].y,pts[s].z);
    glEnd();

    // Tight core: step along each segment at fixed intervals and plot a point.
    // This is how Bresenham works — advance by one unit each step, no division inside.
    const float STEP=0.18f;
    glPointSize(3.5f);
    glColor4f(1.f,1.f,.9f, 0.9f);
    glBegin(GL_POINTS);
    for(int s=0;s<segs;s++){
        V3 a=pts[s], b=pts[s+1];
        V3 dir=(b-a).norm();
        float len=(b-a).len();
        for(float t=0;t<len;t+=STEP){
            V3 q=a+dir*t;
            glVertex3f(q.x,q.y,q.z);
        }
    }
    glEnd();
    glLineWidth(1.f);
    glPointSize(1.f);

    for(auto& p:g_pillars){
        if(!p.crystalLit) continue;
        float s=.18f+.10f*sinf(g_beamGlow*5.2f);
        glColor4f(1.f,.9f,.3f,alpha*.65f);
        glPushMatrix();
        glTranslatef(p.pos.x,p.pos.y+3.12f,p.pos.z);
        glBegin(GL_QUADS);
        glVertex3f(-s,-s,0); glVertex3f(s,-s,0);
        glVertex3f(s,s,0);   glVertex3f(-s,s,0);
        glEnd();
        glPopMatrix();
    }

    if(g_puzzleDone){
        float pulse=.5f+.5f*sinf(g_time*3.8f);
        glPushMatrix(); glTranslatef(0,1.88f,-8.5f);
        for(int ring=0;ring<4;ring++){
            float rr=(.55f+.30f*pulse)*(1.f+ring*.45f);
            glColor4f(1.f,.82f,.18f, .40f/((float)ring+1)*pulse);
            glBegin(GL_LINE_LOOP);
            for(int k=0;k<36;k++){
                float a=k*2.f*3.14159f/36;
                glVertex3f(cosf(a)*rr,sinf(a)*rr,0);
            }
            glEnd();
        }
        glPopMatrix();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

// Torch flame — a fixed double cone sitting on the wall bracket.
// The flicker value (owned by Member 4) changes only the colour, not the shape or position.
// This makes it look like a real wall torch that stays put and brightens/dims with the fire.
static void drawTorchFlame(float x, float y, float z, float flicker){
    glPushMatrix();
    glTranslatef(x, y, z);

    // Outer body — orange, dims when flicker is low
    glColor3f(1.0f * flicker, 0.38f * flicker, 0.0f);
    drawCone(0.07f, 0.26f, 10);

    // Bright inner core — yellow-white at the tip, also flicker-driven
    glPushMatrix();
    glTranslatef(0, 0.06f, 0);
    glColor3f(1.0f * flicker, 0.80f * flicker, 0.15f * flicker);
    drawCone(0.032f, 0.20f, 10);
    glPopMatrix();

    glPopMatrix();
}

static void drawParticles(){
    if(g_particles.empty()) return;
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    glDepthMask(GL_FALSE);
    for(auto& p:g_particles){
        float lr=CLAMP(p.life/p.maxLife,0,1);
        float s=p.size*lr;
        glColor4f(p.r,p.g,p.b,p.a*lr);
        glPushMatrix();
        glTranslatef(p.pos.x,p.pos.y,p.pos.z);
        glBegin(GL_QUADS);
        glVertex3f(-s,-s,0);glVertex3f(s,-s,0);
        glVertex3f(s,s,0);glVertex3f(-s,s,0);
        glEnd();
        glPopMatrix();
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

static void drawHeart(float x,float y,float sz,bool filled){
    glPushMatrix(); glTranslatef(x,y,0);
    glColor4f(filled?.9f:.30f, filled?.12f:.08f, filled?.12f:.08f, 1.f);
    int N=20;
    for(int h=0;h<2;h++){
        float ox=(h==0?-sz*.22f:sz*.22f);
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(ox,sz*.12f);
        for(int k=0;k<=N;k++){
            float a=-3.14159f+k*2.f*3.14159f/N;
            glVertex2f(ox+cosf(a)*sz*.28f, sz*.12f+sinf(a)*sz*.28f);
        }
        glEnd();
    }
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(0,-sz*.5f);
    glVertex2f(-sz*.48f,sz*.22f);
    glVertex2f(sz*.48f,sz*.22f);
    glEnd();
    glPopMatrix();
}

// HUD drawn in 2D screen space. Projection switches to gluOrtho2D so vertex
// coordinates map directly to pixels, then restores perspective when done.
static void drawHUD(){
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix(); glLoadIdentity();
    gluOrtho2D(0,SCR_W,SCR_H,0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix(); glLoadIdentity();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(0.f,0.f,0.f,.50f);
    glBegin(GL_QUADS);
    glVertex2f(0,0);glVertex2f(200,0);
    glVertex2f(200,62);glVertex2f(0,62);
    glEnd();
    for(int i=0;i<MAX_LIVES;i++)
        drawHeart(28.f+i*52.f, 32.f, 19.f, i<g_lives);

    glColor4f(0.f,0.f,0.f,.55f);
    glBegin(GL_QUADS);
    glVertex2f(0,SCR_H-30);glVertex2f(SCR_W,SCR_H-30);
    glVertex2f(SCR_W,SCR_H);glVertex2f(0,SCR_H);
    glEnd();
    glColor3f(.80f,.70f,.30f);
    glRasterPos2f(8,SCR_H-11);
    const char* ctrl="WASD / Arrow keys: Move   |   Mouse: Look   |   TAB: Select pillar   |   Q/E: Rotate (must be nearby!)   |   R: Reset   |   H: Help";
    for(const char* c=ctrl;*c;c++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,c[0]);

    glColor4f(0.f,0.f,0.f,.52f);
    glBegin(GL_QUADS);
    glVertex2f(SCR_W-220,0);glVertex2f(SCR_W,0);
    glVertex2f(SCR_W,160);glVertex2f(SCR_W-220,160);
    glEnd();

    glColor3f(.85f,.72f,.28f);
    glRasterPos2f(SCR_W-212,18);
    const char* ph="~ BEAM STATUS ~";
    for(const char* c=ph;*c;c++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,c[0]);

    for(int i=0;i<(int)g_pillars.size();i++){
        bool lit=g_pillars[i].crystalLit;
        bool near2=(g_nearPillar==i);
        if(lit)        glColor3f(.9f,.8f,.15f);
        else if(near2) glColor3f(.3f,.9f,.4f);
        else           glColor3f(.6f,.5f,.4f);
        char buf[64];
        sprintf(buf," P%d: %s  [%.0f deg]",
            i+1, lit?"LIT  ":"unlit",
            fmodf(g_pillars[i].rotY+360.f,360.f));
        glRasterPos2f(SCR_W-212, 36+i*22);
        for(const char* c=buf;*c;c++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,c[0]);
    }

    glColor3f(.4f,.9f,.5f);
    char selbuf[64];
    sprintf(selbuf," Selected: Pillar %d", g_selPillar+1);
    glRasterPos2f(SCR_W-212, 104);
    for(const char* c=selbuf;*c;c++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,c[0]);

    if(g_nearPillar>=0){
        glColor3f(.3f,1.f,.5f);
        char nbuf[128];
        if (g_nearPillar < 3) {
            sprintf(nbuf," [NEAR P%d — Q/E to rotate]", g_nearPillar+1);
        } else if (g_nearPillar == 3) {
            if (g_puzzleDone) sprintf(nbuf," [NEAR WHEEL — Q/E to crank door]");
            else sprintf(nbuf," [WHEEL LOCKED — solve pillars first]");
        } else if (g_nearPillar == 4) {
            sprintf(nbuf," [NEAR CHEST — Press E to open]");
        }
        glRasterPos2f(SCR_W-212,124);
        for(const char* c=nbuf;*c;c++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,c[0]);
    } else {
        glColor3f(.8f,.4f,.3f);
        glRasterPos2f(SCR_W-212,124);
        const char* far2=" [too far — move closer]";
        for(const char* c=far2;*c;c++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,c[0]);
    }

    if(g_dmgCooldown){
        float a=CLAMP(g_dmgTimer/.6f,0,1)*.45f;
        glColor4f(.9f,.05f,.05f,a);
        glBegin(GL_QUADS);
        glVertex2f(0,0);glVertex2f(SCR_W,0);
        glVertex2f(SCR_W,SCR_H);glVertex2f(0,SCR_H);
        glEnd();
    }

    if(g_msgTimer>0.f){
        float al=CLAMP(g_msgTimer,0.f,1.f);
        glColor4f(0,0,0,.65f*al);
        glBegin(GL_QUADS);
        glVertex2f(SCR_W/2-300,SCR_H/2-32);
        glVertex2f(SCR_W/2+300,SCR_H/2-32);
        glVertex2f(SCR_W/2+300,SCR_H/2+24);
        glVertex2f(SCR_W/2-300,SCR_H/2+24);
        glEnd();
        glColor4f(1.f,.90f,.28f,al);
        float tw=g_msg.size()*8.f;
        glRasterPos2f(SCR_W/2-tw/2, SCR_H/2+8);
        for(char c:g_msg) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,c);
    }

    if(g_showHelp){
        const float BX0 = SCR_W/2 - 290.f;
        const float BX1 = SCR_W/2 + 290.f;
        const float BY0 = 82.f;
        const float LH  = 15.f;
        const float PAD = 10.f;
        const char* lines[]={
            "======  HOW TO WIN  ======",
            "",
            "GOAL: Guide the light beam through all 3 pillars",
            "      to unlock the Wheel Pillar at the back.",
            "",
            "STEP 1  Walk close to PILLAR 1 (green arrow above it).",
            "        Crystal turns GOLD when correct (target: 90 deg).",
            "",
            "STEP 2  Navigate past the crushers to PILLAR 2.",
            "        Rotate until gold  (target: 135 deg).",
            "",
            "STEP 3  Pass the spike row and reach PILLAR 3.",
            "        Rotate until gold  (target: 315 deg).",
            "",
            "STEP 4  Turn the Wheel Pillar to open the Great Door.",
            "        Enter the secret room and press E near the chest!",
            "",
            "HAZARDS: Crushers / Spikes / Arrows each cost 1 life.",
            "Lose all 3 lives = full reset.  H to close this."
        };
        int nlines = (int)(sizeof(lines)/sizeof(lines[0]));
        const float BY1 = BY0 + PAD + nlines * LH + PAD;

        glColor4f(.04f,.03f,.02f,.90f);
        glBegin(GL_QUADS);
        glVertex2f(BX0,BY0); glVertex2f(BX1,BY0);
        glVertex2f(BX1,BY1); glVertex2f(BX0,BY1);
        glEnd();
        glColor4f(.8f,.62f,.18f,.85f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(BX0,BY0); glVertex2f(BX1,BY0);
        glVertex2f(BX1,BY1); glVertex2f(BX0,BY1);
        glEnd();
        glColor4f(.5f,.38f,.08f,.50f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(BX0+3,BY0+3); glVertex2f(BX1-3,BY0+3);
        glVertex2f(BX1-3,BY1-3); glVertex2f(BX0+3,BY1-3);
        glEnd();

        for(int i=0;i<nlines;i++){
            if(i==0)                  glColor4f(1.f,.88f,.22f,1.f);
            else if(lines[i][0]=='S') glColor4f(.55f,.95f,.55f,1.f);
            else                      glColor4f(.88f,.82f,.68f,1.f);
            float ty = BY0 + PAD + (i+1)*LH;
            glRasterPos2f(BX0 + PAD, ty);
            for(const char* c=lines[i];*c;c++)
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c[0]);
        }
    }

    if(g_phase==PHASE_WIN){
        float pulse   = 0.55f + 0.45f * sinf(g_time * 2.8f);
        float pulse2  = 0.60f + 0.40f * sinf(g_time * 3.5f + 1.f);

        glColor4f(.08f,.06f,.01f,.72f);
        glBegin(GL_QUADS);
        glVertex2f(0,0);glVertex2f(SCR_W,0);
        glVertex2f(SCR_W,SCR_H);glVertex2f(0,SCR_H);
        glEnd();

        const float PW = 560.f, PH = 220.f;
        const float PX = SCR_W/2 - PW/2, PY = SCR_H/2 - PH/2;

        glColor4f(.10f,.08f,.02f,.96f);
        glBegin(GL_QUADS);
        glVertex2f(PX,PY);       glVertex2f(PX+PW,PY);
        glVertex2f(PX+PW,PY+PH);glVertex2f(PX,PY+PH);
        glEnd();

        glColor4f(1.f,.82f,.12f, pulse * .9f);
        glLineWidth(3.f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(PX,PY);       glVertex2f(PX+PW,PY);
        glVertex2f(PX+PW,PY+PH);glVertex2f(PX,PY+PH);
        glEnd();
        glColor4f(.9f,.65f,.08f, pulse2 * .7f);
        glLineWidth(1.5f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(PX+6,PY+6);       glVertex2f(PX+PW-6,PY+6);
        glVertex2f(PX+PW-6,PY+PH-6);glVertex2f(PX+6,PY+PH-6);
        glEnd();
        glLineWidth(1.f);

        auto drawStar=[&](float cx,float cy){
            glColor4f(1.f,.90f,.25f, pulse*.85f);
            float r1=8.f,r2=4.f;
            glBegin(GL_TRIANGLE_FAN);
            glVertex2f(cx,cy);
            for(int k=0;k<=10;k++){
                float a=k*2.f*3.14159f/10;
                float r=(k%2==0)?r1:r2;
                glVertex2f(cx+cosf(a)*r, cy+sinf(a)*r);
            }
            glEnd();
        };
        drawStar(PX+14,   PY+14);
        drawStar(PX+PW-14,PY+14);
        drawStar(PX+14,   PY+PH-14);
        drawStar(PX+PW-14,PY+PH-14);

        auto centerText=[&](const char* text, void* font, float y){
            int tw = glutBitmapLength(font, (const unsigned char*)text);
            glRasterPos2f(SCR_W/2.f - tw/2.f, y);
            for(const char* c=text;*c;c++) glutBitmapCharacter(font,c[0]);
        };

        glColor4f(1.f, .92f, .18f, pulse);
        centerText("CONGRATULATIONS!", GLUT_BITMAP_HELVETICA_18, PY + 46);
        glColor4f(1.f, .80f, .30f, 1.f);
        centerText("You solved the Ancient Temple Puzzle!", GLUT_BITMAP_HELVETICA_12, PY + 76);

        glColor4f(.8f,.6f,.15f,.6f);
        glBegin(GL_LINES);
        glVertex2f(PX+30, PY+92); glVertex2f(PX+PW-30, PY+92);
        glEnd();

        glColor4f(.90f,.82f,.55f,1.f);
        centerText("The altar crystal awakens  -  the treasure is yours!", GLUT_BITMAP_HELVETICA_12, PY + 116);
        glColor4f(.75f,.95f,.45f, pulse2);
        centerText("The chest is open - look inside for the gold!", GLUT_BITMAP_HELVETICA_12, PY + 140);
        glColor4f(1.f, .70f, .20f, pulse2);
        centerText("Press  R  to play again", GLUT_BITMAP_HELVETICA_18, PY + 178);
    }

    if(g_phase==PHASE_DEAD){
        glColor4f(.28f,0.f,0.f,.62f);
        glBegin(GL_QUADS);
        glVertex2f(0,0);glVertex2f(SCR_W,0);
        glVertex2f(SCR_W,SCR_H);glVertex2f(0,SCR_H);
        glEnd();
        glColor3f(1.f,.30f,.30f);
        const char* d1=(g_lives>0)?"YOU HAVE FALLEN...":"GAME OVER - All lives lost!";
        glRasterPos2f(SCR_W/2-strlen(d1)*5.8f,SCR_H/2-22);
        for(const char* c=d1;*c;c++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,c[0]);
        glColor3f(1.f,.62f,.62f);
        const char* d2=(g_lives>0)?"Respawning - your pillar progress is saved...":"Press R to start over from the beginning.";
        glRasterPos2f(SCR_W/2-strlen(d2)*4.2f,SCR_H/2+14);
        for(const char* c=d2;*c;c++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,c[0]);
    }

    if(g_phase != PHASE_WIN){
        glDisable(GL_BLEND);
        glColor3f(.95f,.88f,.32f);
        glLineWidth(1.6f);
        glBegin(GL_LINES);
        glVertex2f(SCR_W/2-11,SCR_H/2); glVertex2f(SCR_W/2+11,SCR_H/2);
        glVertex2f(SCR_W/2,SCR_H/2-11); glVertex2f(SCR_W/2,SCR_H/2+11);
        glEnd();
        glLineWidth(1.f);
    }

    glMatrixMode(GL_PROJECTION); glPopMatrix();
    glMatrixMode(GL_MODELVIEW);  glPopMatrix();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
}

// MEMBER3  Animation — pillar lerp, spike state machine, crusher timer,
//          arrow spawn, particle physics, chest lid, win trigger

void spawnParticles(V3 origin, int n, float r, float g, float b, float spread){
    for(int i=0;i<n;i++){
        Particle p;
        p.pos=origin;
        float ang=((float)rand()/RAND_MAX)*6.28318f;
        float sp=(1.5f+((float)rand()/RAND_MAX)*3.f)*spread;
        float up=1.8f+((float)rand()/RAND_MAX)*2.8f;
        p.vel=V3(cosf(ang)*sp*.4f, up, sinf(ang)*sp*.35f);
        p.r=r; p.g=g; p.b=b; p.a=1.f;
        p.maxLife=p.life=.7f+((float)rand()/RAND_MAX)*1.f;
        p.size=.05f+((float)rand()/RAND_MAX)*.08f;
        g_particles.push_back(p);
    }
}

static void updateAnimation(float dt){
    // Pillar smooth rotation: each frame rotY moves a fraction of the remaining
    // gap toward targetRotY — classic lerp so the crystal glides rather than snaps
    for(auto& p:g_pillars){
        float diff=p.targetRotY-p.rotY;
        while(diff> 180.f) diff-=360.f;
        while(diff<-180.f) diff+=360.f;
        if(fabsf(diff)>.3f) p.rotY+=diff*8.f*dt;
        else p.rotY=p.targetRotY;
    }

    // Spike rows alternate between fully up and fully down.
    // Row 2 is offset by inverting the lerp weights so they never both peak together.
    if(g_spikesUp){
        g_spikeOffset += 0.8f * dt;
        if(g_spikeOffset >= 1.0f){ g_spikeOffset = 1.0f; g_spikesUp = false; }
    } else {
        g_spikeOffset -= 0.8f * dt;
        if(g_spikeOffset <= 0.0f){ g_spikeOffset = 0.0f; g_spikesUp = true; }
    }

    // Arrows fire from wall slits every few seconds, random side, random speed
    g_arrowTimer-=dt;
    if(g_arrowTimer<=0.f && g_phase==PHASE_EXPLORE){
        g_arrowTimer=3.f+((float)rand()/RAND_MAX)*2.f;
        for(float az:{-3.5f,-.5f}){
            int side=(rand()%2)*2-1;
            Arrow ar;
            ar.pos=V3(side*(ROOM_W*.5f-.18f), 2.0f, az);
            ar.dir=V3((float)-side, 0, 0);
            ar.speed=4.5f+((float)rand()/RAND_MAX)*3.f;
            ar.life=3.5f; ar.active=true;
            g_arrows.push_back(ar);
        }
    }
    for(auto& ar:g_arrows){
        if(!ar.active) continue;
        ar.pos+=ar.dir*ar.speed*dt;
        ar.life-=dt;
        if(ar.life<=0.f) ar.active=false;
    }
    g_arrows.erase(std::remove_if(g_arrows.begin(),g_arrows.end(),
        [](const Arrow& a){return !a.active;}), g_arrows.end());

    // Particle step: position += velocity * dt, gravity pulls vy down, floor bounce
    for(auto& p:g_particles){
        p.pos+=p.vel*dt;
        p.vel.y-=3.8f*dt;
        if(p.pos.y<.05f && p.vel.y<0){ p.pos.y=.05f; p.vel.y*=-.25f; }
        p.life-=dt;
    }
    g_particles.erase(std::remove_if(g_particles.begin(),g_particles.end(),
        [](const Particle& p){return p.life<=0.f;}), g_particles.end());

    // Chest lid lerps from 0 to 90 degrees when g_chestOpening is set.
    // Win state triggers once the lid has fully settled at 90.
    float cTarget=g_chestOpening ? 90.f : 0.f;
    float cdiff=cTarget-g_chestLid;
    if(fabsf(cdiff)>.08f) g_chestLid+=cdiff*2.2f*dt;
    else {
        g_chestLid=cTarget;
        if (g_chestOpening && g_phase != PHASE_WIN) {
            g_phase = PHASE_WIN;
            spawnParticles(V3(0,.80f,-15.0f),  120, 1.f,.85f,.10f);
            spawnParticles(V3(0,.80f,-15.0f),   40, 1.f,.65f,.05f, 1.4f);
            g_arrowTimer = 9999.f;
            g_msgTimer   = 0.f;
        }
    }

    if(g_phase == PHASE_WIN){
        g_solveTimer+=dt;
        if(g_solveTimer>1.2f){
            spawnParticles(V3(0,.80f,-15.0f), 35, 1.f,.82f,.10f, 1.1f);
            spawnParticles(V3(0,1.38f,-8.5f),15, 1.f,.70f,.15f, .6f);
            g_solveTimer=0.f;
        }
    }

    if(g_msgTimer>0.f) g_msgTimer-=dt;

    if(g_dmgCooldown){
        g_dmgTimer-=dt;
        if(g_dmgTimer<=0.f){ g_dmgCooldown=false; g_dmgTimer=0.f; }
    }
}

// MEMBER4  Lighting — Phong point lights for all torches, spotlight on beam
//          emitter, win flash light, multi-frequency flicker algorithm

static void setupLighting(){
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);
    float ga[]={.08f,.06f,.04f,1.f};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT,ga);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,GL_TRUE);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
    for(int i=0;i<8;i++) glEnable(GL_LIGHT0+i);
    GLfloat spec[]={.6f,.5f,.35f,1.f};
    GLfloat shin[]={32.f};
    glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,spec);
    glMaterialfv(GL_FRONT_AND_BACK,GL_SHININESS,shin);
}

static void initTorches(){
    float hw=ROOM_W*.5f;
    float zz[]={-6.f,-2.f,2.f,6.f};
    for(float z:zz){
        g_torches.push_back({V3(-hw+.12f,3.35f,z),1.f});
        g_torches.push_back({V3( hw-.12f,3.35f,z),1.f});
    }
}

static void updateTorchFlicker(float dt){
    (void)dt;
    // Three overlapping sine waves at different frequencies give a natural
    // flame feel — no two periods align, so the pattern never looks mechanical.
    for(int i=0;i<(int)g_torches.size();i++){
        float base = .70f + .18f*sinf(g_time*7.4f  + i*1.87f);
        float micro= .08f * sinf(g_time*23.f + i*3.1f);
        float noise= ((float)rand()/RAND_MAX-.5f)*.06f;
        g_torches[i].flicker=CLAMP(base+micro+noise,.28f,1.f);
    }
}

static void applyLighting(){
    // Each torch drives a positional Phong light: warm orange diffuse,
    // tiny ambient, small specular highlight. Quadratic attenuation so
    // the glow fades fast — nearby surfaces are bright, far ones stay dark.
    int lightCount=0;
    for(int i=0;i<(int)g_torches.size()&&lightCount<6;i++,lightCount++){
        GLenum L=GL_LIGHT0+lightCount;
        float f=g_torches[i].flicker;
        V3& p=g_torches[i].pos;
        GLfloat pos[]={p.x,p.y,p.z,1.f};
        GLfloat dif[]={1.2f*f,.62f*f,.14f*f,1.f};
        GLfloat amb[]={.14f*f,.06f*f,.02f*f,1.f};
        GLfloat spc[]={.50f*f,.30f*f,.10f*f,1.f};
        glLightfv(L,GL_POSITION,pos);
        glLightfv(L,GL_DIFFUSE, dif);
        glLightfv(L,GL_AMBIENT, amb);
        glLightfv(L,GL_SPECULAR,spc);
        glLightf(L,GL_CONSTANT_ATTENUATION, .25f);
        glLightf(L,GL_LINEAR_ATTENUATION,   .08f);
        glLightf(L,GL_QUADRATIC_ATTENUATION,.025f);
    }

    // Spotlight follows the beam emitter direction toward pillar 1.
    // GL_SPOT_CUTOFF limits the cone to 22 degrees so it only illuminates
    // geometry along the actual beam path.
    {
        GLenum SL=GL_LIGHT6;
        GLfloat pos[]={-ROOM_W*.5f+.25f,3.1f,0.f,1.f};
        V3 dir(1,0,0);
        if(!g_pillars.empty()){
            V3 src(-ROOM_W*.5f+.25f,3.1f,0.f);
            dir=(g_pillars[0].pos+V3(0,3.12f,0)-src).norm();
        }
        GLfloat sd[]={dir.x,dir.y,dir.z};
        float intensity=g_puzzleDone?3.f:(.8f+.35f*sinf(g_beamGlow*3.f));
        GLfloat dif[]={.9f*intensity,.88f*intensity,.30f*intensity,1.f};
        GLfloat spc[]={1.f,.9f,.4f,1.f};
        GLfloat amb[]={0,0,0,1};
        glLightfv(SL,GL_POSITION,      pos);
        glLightfv(SL,GL_DIFFUSE,       dif);
        glLightfv(SL,GL_SPECULAR,      spc);
        glLightfv(SL,GL_AMBIENT,       amb);
        glLightfv(SL,GL_SPOT_DIRECTION,sd);
        glLightf(SL,GL_SPOT_CUTOFF,    22.f);
        glLightf(SL,GL_SPOT_EXPONENT,  8.f);
        glLightf(SL,GL_CONSTANT_ATTENUATION, .25f);
        glLightf(SL,GL_LINEAR_ATTENUATION,   .03f);
        glLightf(SL,GL_QUADRATIC_ATTENUATION,.006f);
    }

    // Victory light at the wheel pedestal — only on when puzzle is done,
    // pulses at 4.2 Hz to match the coin burst rhythm
    {
        GLenum WL=GL_LIGHT7;
        float intensity = g_puzzleDone ? (.65f+.35f*sinf(g_time*4.2f)) : .0f;
        GLfloat pos[]={0.f,2.5f,-8.5f,1.f};
        GLfloat dif[]={1.f*intensity,.88f*intensity,.14f*intensity,1.f};
        GLfloat amb[]={0,0,0,1};
        glLightfv(WL,GL_POSITION,pos);
        glLightfv(WL,GL_DIFFUSE, dif);
        glLightfv(WL,GL_AMBIENT, amb);
        glLightf(WL,GL_CONSTANT_ATTENUATION, .2f);
        glLightf(WL,GL_LINEAR_ATTENUATION,   .07f);
        glLightf(WL,GL_QUADRATIC_ATTENUATION,.018f);
    }
}

static void drawTorchFlames(){
    // Flame sits just above the bowl on the sconce bracket.
    // flicker drives colour intensity — see drawTorchFlame in MEMBER2 section.
    for(auto& t:g_torches)
        drawTorchFlame(t.pos.x, t.pos.y+.22f, t.pos.z, t.flicker);
}

// MEMBER5  Transformation — third-person chase camera, mouse look,
//          AABB room clipping, object collision push, proximity gate,
//          beam solve detection, player character model with limb animation

static void applyCamera(){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(65.0,(double)SCR_W/SCR_H,.1,80.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Eye sits behind and above the player — offset along the reverse look direction
    V3 eye = g_camPos - g_camFront * 2.8f + V3(0, 0.6f, 0);

    float hw = ROOM_W * 0.5f - 0.2f;
    float hd = ROOM_D * 0.5f - 0.2f;
    eye.x = CLAMP(eye.x, -hw, hw);
    eye.z = CLAMP(eye.z, -hd, hd);
    eye.y = CLAMP(eye.y, 0.2f, ROOM_H - 0.2f);

    V3 tgt = g_camPos + g_camFront;
    gluLookAt(eye.x, eye.y, eye.z,
              tgt.x, tgt.y, tgt.z,
              0, 1, 0);
}

// Mouse movement updates yaw and pitch Euler angles, which are then
// converted to a direction vector each frame using cos/sin.
// glutWarpPointer recentres the cursor so horizontal rotation has no limit.
static void handleMouse(int x,int y){
    if(g_firstMouse){glutWarpPointer(SCR_W/2,SCR_H/2);g_firstMouse=false;return;}
    if(x == SCR_W/2 && y == SCR_H/2) return;
    float dx=(x-SCR_W/2)*.13f;
    float dy=(SCR_H/2-y)*.13f;
    g_yaw  +=dx;
    g_pitch =CLAMP(g_pitch+dy,-89.f,89.f);
    g_camFront=V3(
        cosf(RAD(g_yaw))*cosf(RAD(g_pitch)),
        sinf(RAD(g_pitch)),
        sinf(RAD(g_yaw))*cosf(RAD(g_pitch))
    ).norm();
    glutWarpPointer(SCR_W/2,SCR_H/2);
}

void updateNearestPillar(){
    g_nearPillar=-1;
    g_nearDist=999.f;
    for(int i=0;i<(int)g_pillars.size();i++){
        V3 d=g_camPos-g_pillars[i].pos;
        d.y=0;
        float dist=d.len();
        if(dist<g_nearDist){g_nearDist=dist;g_nearPillar=i;}
    }
    V3 dWheel = g_camPos - V3(0, 0, -8.5f);
    dWheel.y = 0;
    float distWheel = dWheel.len();
    if(distWheel < g_nearDist) { g_nearDist = distWheel; g_nearPillar = 3; }

    V3 dChest = g_camPos - V3(0, 0, -15.0f);
    dChest.y = 0;
    float distChest = dChest.len();
    if(distChest < g_nearDist) { g_nearDist = distChest; g_nearPillar = 4; }

    if(g_nearDist>PILLAR_INTERACT_RADIUS) g_nearPillar=-1;
}

static void rotatePillar(int idx, float dir){
    if(idx<0||idx>=(int)g_pillars.size()) return;
    bool inRange=(g_nearPillar==idx);
    if(!inRange){
        showMessage("Get closer to the pillar first!", 1.8f);
        return;
    }
    g_pillars[idx].targetRotY+=dir*SNAP_STEP;
    spawnParticles(V3(g_pillars[idx].pos.x,2.5f,g_pillars[idx].pos.z),
                   6,.9f,.7f,.2f,.4f);
    bool solved=checkBeamSolved();
    if(solved&&!g_puzzleDone){
        g_puzzleDone=true;
        showMessage("The Wheel Pillar unlocks! Turn it to open the door.", 3.f);
        spawnParticles(V3(0,1.38f,-8.5f),  50, 1.f,.88f,.20f, .7f);
        spawnParticles(V3(0,1.38f,-8.5f),  20, .8f,.4f,.1f,   .4f);
    }
    if(!solved && g_puzzleDone && g_phase != PHASE_WIN){
        g_puzzleDone=false;
        showMessage("A pillar moved! The Wheel locks again.", 2.5f);
    }
}

// Sequential beam check: each pillar's rotY is compared to SOLUTION[i].
// Pillar i only counts as lit if all pillars before it are also lit —
// so the beam has to propagate in order from the emitter end.
bool checkBeamSolved(){
    auto angNorm=[](float a)->float{
        a=fmodf(a,360.f);
        return a<0?a+360.f:a;
    };
    for(int i=0;i<3;i++){
        float cur=angNorm(g_pillars[i].targetRotY);
        float sol=SOLUTION[i];
        float diff=fabsf(cur-sol);
        if(diff>180.f) diff=360.f-diff;
        g_pillars[i].crystalLit=(diff<=SOLVE_TOL);
    }
    g_pillars[0].showGuide=true;
    g_pillars[1].showGuide=g_pillars[0].crystalLit;
    g_pillars[2].showGuide=g_pillars[0].crystalLit&&g_pillars[1].crystalLit;
    if(!g_pillars[0].crystalLit){g_pillars[1].crystalLit=false;g_pillars[2].crystalLit=false;}
    else if(!g_pillars[1].crystalLit){g_pillars[2].crystalLit=false;}
    return g_pillars[0].crystalLit&&g_pillars[1].crystalLit&&g_pillars[2].crystalLit;
}

void showMessage(const std::string& s,float dur){g_msg=s;g_msgTimer=dur;}

void respawn(){
    g_camPos=g_spawnPos;
    g_phase=PHASE_EXPLORE;
    g_firstMouse=true;
    g_dmgCooldown=true;
    g_dmgTimer=1.5f;
    glutWarpPointer(SCR_W/2,SCR_H/2);
}

void checkTrapCollisions(){
    if(g_phase!=PHASE_EXPLORE || g_dmgCooldown) return;
    V3 cp=g_camPos;

    auto takeDamage=[&](const char* msg){
        g_lives--;
        spawnParticles(cp,12,1.f,.18f,.18f,.5f);
        if(g_lives<=0){
            g_lives=0; g_phase=PHASE_DEAD; g_deadTimer=3.f;
            char buf[120];
            sprintf(buf,"%s  GAME OVER - press R to restart.",msg);
            showMessage(buf,3.f);
        } else {
            g_phase=PHASE_DEAD; g_deadTimer=2.f;
            char buf[120];
            sprintf(buf,"%s  Lives left: %d - respawning...",msg,g_lives);
            showMessage(buf,2.f);
        }
    };

    float y1 = LERP(-0.95f, -0.05f, g_spikeOffset);
    float y2 = LERP(-0.05f, -0.95f, g_spikeOffset);

    if(cp.z>2.0f&&cp.z<3.2f&&fabsf(cp.x)<3.5f&&y1>-0.4f)
        takeDamage("Spikes!");

    if(cp.z>-5.2f&&cp.z<-3.8f&&fabsf(cp.x-1.5f)<2.5f&&y2>-0.4f)
        takeDamage("Spikes from below!");

    if(isFloorHole(cp.x, cp.z))
        takeDamage("Fell into the pit!");

    struct CrusherTrap { float x, z, phaseOff; };
    CrusherTrap crushers[]={ {-0.8f, .8f, 4.8f}, {3.2f, -1.2f, 0.33f}, {0.0f, -5.5f, 0.66f} };
    for(auto& c:crushers){
        float phase = fmodf(g_time * 0.4f + c.phaseOff, 1.0f);
        float h = ROOM_H;
        if (phase < 0.1f) h = ROOM_H - (ROOM_H * (phase * 10.0f));
        else if (phase < 0.3f) h = 0.0f;
        else h = (phase - 0.3f) / 0.7f * ROOM_H;
        if (h < 1.6f) {
            if (fabsf(cp.x - c.x) < 0.7f && fabsf(cp.z - c.z) < 0.7f) {
                takeDamage("Crushed!");
            }
        }
    }

    for(auto& ar:g_arrows){
        if(!ar.active) continue;
        V3 diff=cp-ar.pos;
        if(diff.len()<0.46f){
            ar.active=false;
            takeDamage("Arrow!");
        }
    }
}

// Camera movement: projects forward/right onto the horizontal plane so
// the player can't fly, then clamps position inside room bounds (AABB clipping).
// Also pushes the player out of any overlapping solid objects.
static void moveCamera(V3 delta){
    if (g_camPos.y < 1.0f) return;
    g_camPos+=delta;
    g_walkPhase += delta.len() * 10.0f;
    float hw=ROOM_W*.5f-.35f, hd=ROOM_D*.5f-.35f;
    g_camPos.x=CLAMP(g_camPos.x,-hw,hw);
    g_camPos.z=CLAMP(g_camPos.z,-hd,hd);

    float pr = 0.35f;
    for(const auto& p : g_pillars) {
        float minDist = pr + 0.5f;
        V3 diff = g_camPos - p.pos;
        diff.y = 0;
        float dist = diff.len();
        if(dist > 0.001f && dist < minDist)
            g_camPos += diff.norm() * (minDist - dist);
    }
    {
        V3 altarPos(0.0f, 0.0f, -8.5f);
        float minDist = pr + 0.8f;
        V3 diff = g_camPos - altarPos;
        diff.y = 0;
        float dist = diff.len();
        if(dist > 0.001f && dist < minDist)
            g_camPos += diff.norm() * (minDist - dist);
    }
    {
        V3 chestPos(0.0f, 0.0f, -15.0f);
        float minDist = pr + 0.6f;
        V3 diff = g_camPos - chestPos;
        diff.y = 0;
        float dist = diff.len();
        if(dist > 0.001f && dist < minDist)
            g_camPos += diff.norm() * (minDist - dist);
    }
    if (g_camPos.z < -9.6f && g_camPos.z > -10.4f) {
        bool hitWall = fabsf(g_camPos.x) > 1.4f;
        bool hitDoor = g_doorY < 2.0f;
        if (hitWall || hitDoor) {
            if (g_camPos.z - delta.z >= -9.6f) g_camPos.z = -9.6f;
            else if (g_camPos.z - delta.z <= -10.4f) g_camPos.z = -10.4f;
            else g_camPos.z = -9.6f;
        }
    }

    g_camPos.y=1.8f;
    updateNearestPillar();
    checkTrapCollisions();
}

// Player character using hierarchical transforms.
// Each limb rotates around its joint pivot using glPushMatrix + glRotate + offset.
// Arms and legs swing on a sin wave driven by distance walked (g_walkPhase).
static void drawPlayerCharacter() {
    glPushMatrix();
    glTranslatef(g_camPos.x, 0.0f, g_camPos.z);
    glRotatef(-g_yaw - 90.0f, 0, 1, 0);

    float armSwing = sinf(g_walkPhase) * 30.0f;
    float legSwing = sinf(g_walkPhase) * 30.0f;

    glColor3f(0.25f, 0.2f, 0.15f);
    glPushMatrix();
    glTranslatef(-0.12f, 0.5f, 0);
    glRotatef(-legSwing, 1, 0, 0);
    glTranslatef(0, -0.25f, 0);
    drawBox(0.16f, 0.5f, 0.16f);
    glPopMatrix();

    glPushMatrix();
    glTranslatef( 0.12f, 0.5f, 0);
    glRotatef(legSwing, 1, 0, 0);
    glTranslatef(0, -0.25f, 0);
    drawBox(0.16f, 0.5f, 0.16f);
    glPopMatrix();

    float bob = sinf(g_time * 5.0f) * 0.04f;
    glTranslatef(0, bob, 0);

    glColor3f(0.15f, 0.4f, 0.3f);
    glPushMatrix(); glTranslatef(0, 0.9f, 0); drawBox(0.45f, 0.7f, 0.25f); glPopMatrix();

    glColor3f(0.85f, 0.65f, 0.5f);
    glPushMatrix(); glTranslatef(0, 1.45f, 0); drawSphere(0.2f, 12, 12); glPopMatrix();

    glColor3f(0.15f, 0.4f, 0.3f);
    glPushMatrix();
    glTranslatef(-0.3f, 1.15f, 0);
    glRotatef(armSwing, 1, 0, 0);
    glTranslatef(0, -0.15f, 0);
    drawBox(0.15f, 0.5f, 0.15f);
    glPopMatrix();

    glPushMatrix();
    glTranslatef( 0.3f, 1.15f, 0);
    glRotatef(-armSwing, 1, 0, 0);
    glTranslatef(0, -0.15f, 0);
    drawBox(0.15f, 0.5f, 0.15f);
    glPopMatrix();

    glColor3f(0.35f, 0.2f, 0.1f);
    glPushMatrix(); glTranslatef(0, 0.9f, 0.18f); drawBox(0.35f, 0.5f, 0.15f); glPopMatrix();

    glPopMatrix();
}

static void display(){
    glClearColor(.018f,.012f,.008f,1.f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    applyCamera();
    applyLighting();
    drawRoom();
    drawPillars();
    drawWheelPillar();
    drawTreasureChest();
    drawTraps();
    drawPlayerCharacter();
    drawTorchFlames();
    drawBeamGlow();
    drawParticles();
    drawHUD();
    glutSwapBuffers();
}

static void reshape(int w,int h){
    if(h==0)h=1;
    glViewport(0,0,w,h);
}

static void doFullReset(){
    g_lives=MAX_LIVES; g_puzzleDone=false;
    g_phase=PHASE_EXPLORE; g_chestLid=0.f;
    g_chestOpening=false;
    g_wheelRot=0.f; g_doorY=0.f;
    g_particles.clear(); g_arrows.clear();
    g_solveTimer=0.f; g_msgTimer=0.f;
    g_arrowTimer=2.f;
    g_dmgCooldown=false;
    for(auto& p:g_pillars){
        p.rotY=p.targetRotY=0.f;
        p.crystalLit=false; p.showGuide=false;
    }
    checkBeamSolved();
    respawn();
    showMessage("Temple reset - align the beam to win!",3.f);
}

static void keyboard(unsigned char key,int,int){
    if(g_phase==PHASE_WIN){
        if(key=='r'||key=='R') doFullReset();
        if(key==27) exit(0);
        return;
    }
    if(g_phase==PHASE_DEAD && g_lives<=0){
        if(key=='r'||key=='R') doFullReset();
        if(key==27) exit(0);
        return;
    }
    {
        float spd=.20f;
        V3 right=g_camFront.cross(V3(0,1,0)).norm();
        V3 fwd(g_camFront.x,0,g_camFront.z); fwd=fwd.norm();
        switch(key){
            case'w':case'W': moveCamera(fwd*spd);  break;
            case's':case'S': moveCamera(fwd*-spd); break;
            case'a':case'A': moveCamera(right*-spd);break;
            case'd':case'D': moveCamera(right*spd); break;
            case 9:
                g_selPillar=(g_selPillar+1)%(int)g_pillars.size();
                { char buf[60]; sprintf(buf,"Pillar %d selected",g_selPillar+1);
                  showMessage(buf,1.2f); }
                break;
            case'q':case'Q':
                if (g_nearPillar >= 0 && g_nearPillar < 3) rotatePillar(g_nearPillar, -1.f);
                else if (g_nearPillar == 3 && g_puzzleDone) { g_wheelRot -= 45.f; g_doorY = CLAMP(g_doorY + 0.4f, 0.f, 4.0f); }
                break;
            case'e':case'E':
                if (g_nearPillar >= 0 && g_nearPillar < 3) rotatePillar(g_nearPillar, 1.f);
                else if (g_nearPillar == 3 && g_puzzleDone) { g_wheelRot += 45.f; g_doorY = CLAMP(g_doorY - 0.4f, 0.f, 4.0f); }
                else if (g_nearPillar == 4 && g_doorY > 2.0f) { g_chestOpening = true; }
                break;
            case'o':case'O':
                if (g_nearPillar == 4 && g_doorY > 2.0f) { g_chestOpening = true; }
                break;
            case'h':case'H':
                g_showHelp=!g_showHelp;
                break;
            case 27: exit(0); break;
            case'r':case'R': doFullReset(); break;
        }
    }
}

static void specialKey(int key,int,int){
    if(g_phase==PHASE_WIN) return;
    float spd=.20f;
    V3 right=g_camFront.cross(V3(0,1,0)).norm();
    V3 fwd(g_camFront.x,0,g_camFront.z); fwd=fwd.norm();
    switch(key){
        case GLUT_KEY_UP:    moveCamera(fwd*spd);   break;
        case GLUT_KEY_DOWN:  moveCamera(fwd*-spd);  break;
        case GLUT_KEY_LEFT:  moveCamera(right*-spd);break;
        case GLUT_KEY_RIGHT: moveCamera(right*spd); break;
    }
}

static void mouseMove(int x,int y){handleMouse(x,y);}

static void timerCallback(int){
    float now=(float)glutGet(GLUT_ELAPSED_TIME)/1000.f;
    g_dt=CLAMP(now-g_time,0.f,.05f);
    g_time=now;
    g_beamGlow+=g_dt;

    updateTorchFlicker(g_dt);
    updateAnimation(g_dt);

    if(g_phase==PHASE_DEAD){
        g_deadTimer-=g_dt;
        if(g_deadTimer<=0.f){
            if(g_lives>0) respawn();
        }
    }
    if(g_phase==PHASE_EXPLORE){
        updateNearestPillar();
        checkTrapCollisions();
    }
    checkBeamSolved();

    if(!g_puzzleDone && g_doorY > 0.f && g_phase != PHASE_WIN)
        g_doorY = CLAMP(g_doorY - 1.5f * g_dt, 0.f, 4.0f);

    glutPostRedisplay();
    glutTimerFunc(16,timerCallback,0);
}

int main(int argc,char** argv){
    srand((unsigned)time(nullptr));
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH|GLUT_MULTISAMPLE);
    glutInitWindowSize(SCR_W,SCR_H);
    glutInitWindowPosition(80,60);
    glutCreateWindow("Ancient Temple Puzzle Room - CG Project");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKey);
    glutPassiveMotionFunc(mouseMove);
    glutMotionFunc(mouseMove);
    glutWarpPointer(SCR_W/2,SCR_H/2);
    glutSetCursor(GLUT_CURSOR_NONE);

    glEnable(GL_DEPTH_TEST);
    setupLighting();

    g_pillars.push_back({V3(-2.f, 0, 1.f),   0.f,0.f,false,false,0});
    g_pillars.push_back({V3( 2.5f,0,-2.5f),  0.f,0.f,false,false,1});
    g_pillars.push_back({V3(-1.5f,0,-6.2f),  0.f,0.f,false,false,2});

    checkBeamSolved();
    initTorches();

    showMessage("Align the 3 mirrors to guide the beam to the altar!  Press H for help.",5.f);

    glutTimerFunc(16,timerCallback,0);
    glutMainLoop();
    return 0;
}
