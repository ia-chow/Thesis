{Program to find Besselian day numbers. Adapted from Explanatory
Supplement to the Astronomical Almanac 1992. This program will
find A,B,C,D Besselian Day numbers in arc seconds. A,B,D are
all accurate to more than 0.001 second of arc and C seems to
agree with the Nautical Almanac to within about 0.001 or 0.002
seconds of arc. Note that C and D arte found from the Barycentric
velocity of the Earth and hence the table used here from DE403
may actually be somewhat more accurate than that found in older
nautical almanacs}

{Variable List:
jd - integer Julian day found from fundtion Julday
y - integer year (input from user)
mm - integer month (input from user)
dd - integer day (input from user)
a,b,c,d,e,ss,cc - all are coefficients for the series summations
                to find nutation in longitude and nutation in
                obliquity of the ecliptic
jd1 - actual JD time with decimal for epoch of date
tau - fraction of year measured from epoch of year ie 19xx.5; values are
      negative before middle of year and positive thereafter
deltaphi - Nutation angle in longitude in arc seconds
deltaeps - Nutation angle in obliquity in arc seconds
bea - Besselian Day Number A in arc seconds
beb - Besselian Day Number B in arc seconds
c3  - Besselian Day Number C in arc seconds
d3  - Besselian Day Number D in arc seconds
barytime - JD of lookup table for barycentric velocity components for Earth
xdot - X component of the barycentric velocity of the Earth in AU/day referred to J2000.0
ydot - Y "         "                  "           "            "
zdot - Z "         "                  "           "            "
xdotp -  X component of the barycentric velocity of the Earth in AU/day referred to mean equinox at jd1
ydotp -  Y component of the barycentric velocity of the Earth in AU/day referred to mean equinox at jd1
ibes - internal text file referenced to extrenal file containing terrestrial barycentric velocities
bis - boolean variable which determines if year is a leap year
L - Mean longitude of the moon minus the mean longitude of the moon's perigee
LP - mean longitude of the sun minus the mean longitude of the sun's perigee
F - Mean longitude of the moon minus the mean longitude of the moon's node
DDD - mean elongation of the moon from the sun
OM - longitude of the mean ascending node of the lunar orbit on the ecliptic
     measured from the mean eqinox of date (***NOTE THAT IN THE 1992 VERSION
     OF THE EXPLANATORY SUPPLEMENT TO THE AA, THE LEADING TERM IS IMPROPERLY
     GIVEN AS 135 02' ... AND SHOULD BE 125 02' ... - THIS IS CORRECTED IN
     THIS PROGRAM***)
EO - the mean obliquity of the ecliptic
N - rate per Julian century at reference wpoch of general precession in declination in arc seconds
EPS - Obliquity of the ecliptic at epoch of date
epochjdt - Julian Date for the epoch of year (ie JD at 19xx.5)
ts - difference between jd at epoch of date and fixed epoch in units of Julian centuries
tb - difference between jd at a fixed epoch and the reference epoch (J2000.0) in units of Julian centuries
za - rotation angle in arc seconds for precession
ta - rotation angle in arc seconds for precession
xa - "        "        "                         "
p - precession matrix for epoch of date}


program besseldaynum;
const
   pi=3.14159265358979;
   dtr=0.0174532925199433;
   rtd=57.2957795130823;
var
jd,y,mm,dd:longint;
a,b,c,d,e,ss,cc:array [1..106] of double;
jd1,tau,deltaphi,deltaeps,bea,beb,T:double;
kk,c3,d3,xdotp,ydotp:double;
barytime,xdot,ydot,zdot:array[1..36530] of double;
ibes:text;
finished,bis:boolean;
L,LP,F,DDD,OM,EO,N:double;
eps,epochjdt,ts,tb,za,ta,xa:double;
p:array [1..3,1..3] of double;
question:string[5];
k,j:integer;
i,tref:longint;

{Procedure calculates Julian date for given integer day}
procedure julday(y,mm,dd: longint; var jd:longint);
CONST
   igreg=588829;
VAR
   ja,jm,jy,jul: longint;
BEGIN
   IF  (y = 0) THEN BEGIN
      writeln('there is no year zero.'); readln;
   END;
   IF  (y < 0) THEN  y := y+1;
   IF  (mm > 2)  THEN BEGIN
      jy := y;
      jm := mm+1
   END ELSE BEGIN
      jy := y-1;
      jm := mm+13
   END;
   jul := trunc(365.25*jy)+trunc(30.6001*jm)+dd+1720995;
   IF  (dd+31*(mm+12*y) >= igreg)  THEN BEGIN
      ja := trunc(0.01*jy);
      jul := jul+2-ja+trunc(0.25*ja)
   END;
   jd := jul
END;

begin
{Read in the barycentric velocity of Earth outputted from De403}
reset(ibes,'bes.dat');
for i:=1 to 36525 do
    begin
         readln(ibes,barytime[i],xdot[i],ydot[i],zdot[i]);
    end;
{begin by inputting all constants for the precise calculation of nutation
in longitude and obliquity - see page 112-113 in the 1992 edition of the
Explanatory Supplement}
while not(finished) do
begin
for i:=1 to 106 do
begin
     a[i]:=0; b[i]:=0; c[i]:=0; d[i]:=0; ss[i]:=0; cc[i]:=0;
end;
a[1]:=0.0; b[1]:=0.0; c[1]:=0.0; d[1]:=0.0;  e[1]:=1.0; ss[1]:=-171996.0;
cc[1]:=92025.0; e[2]:=2.0;  ss[2]:=2062.0; cc[2]:=-895.0;
a[3]:=-2.0; c[3]:=2.0;  e[3]:=1.0; ss[3]:=46; cc[3]:=-24;
a[4]:=2;  c[4]:=-2; ss[4]:=11;
a[5]:=-2; c[5]:=2; e[5]:=2; ss[5]:=-3; cc[5]:=1;
a[6]:=1; b[6]:=-1; d[6]:=-1; ss[6]:=-3;
b[7]:=-2; c[7]:=2; d[7]:=-2; e[7]:=1; ss[7]:=-2; cc[7]:=1;
a[8]:=2; c[8]:=-2; e[8]:=1; ss[8]:=1;
c[9]:=2; d[9]:=-2; e[9]:=2; ss[9]:=-13187; cc[9]:=5736;
b[10]:=1; ss[10]:=1426; cc[10]:=54;
b[11]:=1; c[11]:=2; d[11]:=-2; e[11]:=2; ss[11]:=-517; cc[11]:=224;
b[12]:=-1; c[12]:=2; d[12]:=-2; e[12]:=2; ss[12]:=217; cc[12]:=-95;
c[13]:=2; d[13]:=-2; e[13]:=1; ss[13]:=129; cc[13]:=-70;
a[14]:=2; d[14]:=-2; ss[14]:=48; cc[14]:=1;
c[15]:=2; d[15]:=-2; ss[15]:=-22;
b[16]:=2; ss[16]:=17;
b[17]:=1; e[17]:=1; ss[17]:=-15; cc[17]:=9;
b[18]:=2; c[18]:=2; d[18]:=-2; e[18]:=2; ss[18]:=-16; cc[18]:=7;
b[19]:=-1; e[19]:=1; ss[19]:=-12; cc[19]:=6;
a[20]:=-2; d[20]:=2; e[20]:=1; ss[20]:=-6; cc[20]:=3;
b[21]:=-1; c[21]:=2; d[21]:=-2; e[21]:=1; ss[21]:=-5; cc[21]:=3;
a[22]:=2; d[22]:=-2; e[22]:=1; ss[22]:=4; cc[22]:=-2;
b[23]:=1; c[23]:=2; d[23]:=-2; e[23]:=1; ss[23]:=4; cc[23]:=-2;
a[24]:=1; d[24]:=-1; ss[24]:=-4;
a[25]:=2; b[25]:=1; d[25]:=-2; ss[25]:=1;
c[26]:=-2; d[26]:=2; e[26]:=1; ss[26]:=1;
b[27]:=1; c[27]:=-2; d[27]:=2; ss[27]:=-1;
b[28]:=1; e[28]:=2; ss[28]:=1;
a[29]:=-1; d[29]:=1; e[29]:=1; ss[29]:=1;
b[30]:=1; c[30]:=2; d[30]:=-2; ss[30]:=-1;
c[31]:=2; e[31]:=2; ss[31]:=-2274; cc[31]:=977;
a[32]:=1; ss[32]:=712; cc[32]:=-7;
c[33]:=2; e[33]:=1; ss[33]:=-386; cc[33]:=200;
a[34]:=1; c[34]:=2; e[34]:=2; ss[34]:=-301; cc[34]:=129;
a[35]:=1; d[35]:=-2; ss[35]:=-158; cc[35]:=-1;
a[36]:=-1; c[36]:=2; e[36]:=2; ss[36]:=123; cc[36]:=-53;
d[37]:=2; ss[37]:=63; cc[37]:=-2;
a[38]:=1; e[38]:=1; ss[38]:=63; cc[38]:=-33;
a[39]:=-1; e[39]:=1; ss[39]:=-58; cc[39]:=32;
a[40]:=-1; c[40]:=2; d[40]:=2; e[40]:=2; ss[40]:=-59; cc[40]:=26;
a[41]:=1; c[41]:=2; e[41]:=1; ss[41]:=-51; cc[41]:=27;
c[42]:=2; d[42]:=2; e[42]:=2; ss[42]:=-38; cc[42]:=16;
a[43]:=2; ss[43]:=29; cc[43]:=-1;
a[44]:=1; c[44]:=2; d[44]:=-2; e[44]:=2; ss[44]:=29; cc[44]:=-12;
a[45]:=2; c[45]:=2; e[45]:=2; ss[45]:=-31; cc[45]:=13;
c[46]:=2; ss[46]:=26; cc[46]:=-1;
a[47]:=-1; c[47]:=2; e[47]:=1; ss[47]:=21; cc[47]:=-10;
a[48]:=-1; d[48]:=2; e[48]:=1; ss[48]:=16; cc[48]:=-8;
a[49]:=1; d[49]:=-2; e[49]:=1; ss[49]:=-13; cc[49]:=7;
a[50]:=-1; c[50]:=2; d[50]:=2; e[50]:=1; ss[50]:=-10; cc[50]:=5;
a[51]:=1; b[51]:=1; d[51]:=-2; ss[51]:=-7;
b[52]:=1; c[52]:=2; e[52]:=2; ss[52]:=7; cc[52]:=-3;
b[53]:=-1; c[53]:=2; e[53]:=2; ss[53]:=-7; cc[53]:=3;
a[54]:=1; c[54]:=2; d[54]:=2; e[54]:=2; ss[54]:=-8; cc[54]:=3;
a[55]:=1; d[55]:=2; ss[55]:=6;
a[56]:=2; c[56]:=2; d[56]:=-2; e[56]:=2; ss[56]:=6; cc[56]:=-3;
d[57]:=2; e[57]:=1; ss[57]:=-6; cc[57]:=3;
c[58]:=2; d[58]:=2; e[58]:=1; ss[58]:=-7; cc[58]:=3;
a[59]:=1; c[59]:=2; d[59]:=-2; e[59]:=1; ss[59]:=6; cc[59]:=-3;
d[60]:=-2; e[60]:=1; ss[60]:=-5; cc[60]:=3;
a[61]:=1; b[61]:=-1; ss[61]:=5;
a[62]:=2; c[62]:=2; e[62]:=1; ss[62]:=-5; cc[62]:=3;
b[63]:=1; d[63]:=-2; ss[63]:=-4;
a[64]:=1; c[64]:=-2; ss[64]:=4;
d[65]:=1; ss[65]:=-4;
a[66]:=1; b[66]:=1; ss[66]:=-3;
a[67]:=1; c[67]:=2; ss[67]:=3;
a[68]:=1; b[68]:=-1; c[68]:=2; e[68]:=2; ss[68]:=-3; cc[68]:=1;
a[69]:=-1; b[69]:=-1; c[69]:=2; d[69]:=2; e[69]:=2; ss[69]:=-3; cc[69]:=1;
a[70]:=-2; e[70]:=1; ss[70]:=-2; cc[70]:=1;
a[71]:=3; c[71]:=2; e[71]:=2; ss[71]:=-3; cc[71]:=1;
b[72]:=-1; c[72]:=2; d[72]:=2; e[72]:=2; ss[72]:=-3; cc[72]:=1;
a[73]:=1; b[73]:=1; c[73]:=2; e[73]:=2; ss[73]:=2; cc[73]:=-1;
a[74]:=-1; c[74]:=2; d[74]:=-2; e[74]:=1; ss[74]:=-2; cc[74]:=1;
a[75]:=2; e[75]:=1; ss[75]:=2; cc[75]:=-1;
a[76]:=1; e[76]:=2; ss[76]:=-2; cc[76]:=1;
a[77]:=3; ss[77]:=2;
c[78]:=2; d[78]:=1; e[78]:=2; ss[78]:=2; cc[78]:=-1;
a[79]:=-1; e[79]:=2; ss[79]:=1; cc[79]:=-1;
a[80]:=1; d[80]:=-4; ss[80]:=-1;
a[81]:=-2; c[81]:=2; d[81]:=2; e[81]:=2; ss[81]:=1; cc[81]:=-1;
a[82]:=-1; c[82]:=2; d[82]:=4; e[82]:=2; ss[82]:=-2; cc[82]:=1;
a[83]:=2; d[83]:=-4; ss[83]:=-1;
a[84]:=1; b[84]:=1; c[84]:=2; d[84]:=-2; e[84]:=2; ss[84]:=1; cc[84]:=-1;
a[85]:=1; c[85]:=2; d[85]:=2; e[85]:=1; ss[85]:=-1; cc[85]:=1;
a[86]:=-2; c[86]:=2; d[86]:=4; e[86]:=2; ss[86]:=-1; cc[86]:=1;
a[87]:=-1; c[87]:=4; e[87]:=2; ss[87]:=1;
a[88]:=1; b[88]:=-1; d[88]:=-2; ss[88]:=1;
a[89]:=2; c[89]:=2; d[89]:=-2; e[89]:=1; ss[89]:=1; cc[89]:=-1;
a[90]:=2; c[90]:=2; d[90]:=2; e[90]:=2; ss[90]:=-1;
a[91]:=1; d[91]:=2; e[91]:=1; ss[91]:=-1;
c[92]:=4; d[92]:=-2; e[92]:=2; ss[92]:=1;
a[93]:=3; c[93]:=2; d[93]:=-2; e[93]:=2; ss[93]:=1;
a[94]:=1; c[94]:=2; d[94]:=-2; e[94]:=0; ss[94]:=-1;
b[95]:=1; c[95]:=2; e[95]:=1; ss[95]:=1;
a[96]:=-1; b[96]:=-1; d[96]:=2; e[96]:=1; ss[96]:=1;
c[97]:=-2; e[97]:=1; ss[97]:=-1;
c[98]:=2; d[98]:=-1; e[98]:=2; ss[98]:=-1;
b[99]:=1; d[99]:=2; ss[99]:=-1;
a[100]:=1; c[100]:=-2; d[100]:=-2; ss[100]:=-1;
b[101]:=-1; c[101]:=2; e[101]:=1; ss[101]:=-1;
a[102]:=1; b[102]:=1; d[102]:=-2; e[102]:=1; ss[102]:=-1;
a[103]:=1; c[103]:=-2; d[103]:=2; ss[103]:=-1;
a[104]:=2; d[104]:=2; ss[104]:=1;
c[105]:=2; d[105]:=4; e[105]:=2; ss[105]:=-1;
b[106]:=1; d[106]:=1; ss[106]:=1;

{Get input data from the user}
     writeln('What is the year:'); readln(y);
     writeln('What is the month:'); readln(mm);
     writeln('What is the UT day:'); readln(dd);
     t:=0.0;
{Calculate Julian date for given day}
    julday(y,mm,dd,jd);
    t:=t-12;
{Find precise Julian Date with decimal after subtracting 12 hours
since JD's begin at 12 UT (NOT 0 UT)}
    jd1:=jd+t/24;
    T:=(jd1-2451545.0)/36525;
    L:=134.962981+(477000+198.867398)*T+0.0086944*sqr(T)+1.77e-5*T*sqr(T);
    LP:=357.5277233+(35640+359.05034)*T-1.60277e-4*sqr(T)-3.33333e-6*sqr(T)*T;
    F:=93.2719102+(483120+82.017538)*T-0.0036825*sqr(T)+3.0555e-6*sqr(T)*T;
    DDD:=297.850363+(444960+307.11148)*T-0.00191416*sqr(T)+5.278e-6*sqr(T)*T;
    OM:=125.0445222-(1800+134.1362608)*T+2.0708e-3*sqr(T)+2.222e-6*sqr(T)*T;
    EO:=23.4392911-0.0130041*T-1.639E-7*SQR(T)+5.036E-7*SQR(T)*T;
    N:=2004.3109-0.85330*T-0.000217*sqr(T);
{Make time corrections for nutation components in fourier series}
      ss[1]:=ss[1]-174.2*T; cc[1]:=cc[1]+8.9*T;
      ss[2]:=ss[2]+0.2*T; cc[2]:=cc[2]+0.5*T;
      ss[9]:=ss[9]-1.6*T; cc[9]:=cc[9]-3.1*T;
      ss[10]:=ss[10]-3.4*T; cc[10]:=cc[10]-0.1*T;
      ss[11]:=ss[11]+1.2*T; cc[11]:=cc[11]-0.6*T;
      ss[12]:=ss[12]-0.5*T; cc[12]:=cc[12]+0.3*T;
      ss[13]:=ss[13]+0.1*T;
      ss[16]:=ss[16]-0.1*T;
      ss[18]:=ss[18]+0.1*T;
      ss[31]:=ss[31]-0.2*T; cc[31]:=cc[31]-0.5*T;
      ss[32]:=ss[32]+0.1*T;
      ss[33]:=ss[33]-0.4*T;
      cc[34]:=cc[34]-0.1*T;
      ss[38]:=ss[38]+0.1*T;
      ss[39]:=ss[39]-0.1*T;
      deltaphi:=0.0;
      deltaeps:=0.0;
{Now sum coefficients for nutation angles and find the angles}
for i:=1 to 106 do
    begin
         ss[i]:=ss[i]*0.0001; cc[i]:=cc[i]*0.0001;
         deltaphi:=deltaphi+ss[i]*sin((a[i]*L+b[i]*LP+c[i]*F+d[i]*DDD+e[i]*OM)*dtr);
         deltaeps:=deltaeps+cc[i]*cos((a[i]*L+b[i]*LP+c[i]*F+d[i]*DDD+e[i]*OM)*dtr);
    end;
    julday(y,1,1,jd);
{Determine if this year is a leap year}
    bis:=false;
    if ((y mod 4)=0) then bis:=true;
    if ((y mod 100)=0) then bis:=false;
    if ((y mod 400)=0) then bis:=true;
{Calculate the time relative to the epoch of year in the middle of the year}
if (bis) then tau:=(jd1-jd+182.625)/365.25-1
   else tau:=(jd1-jd+183.125)/365.25-1;
{Recall that we need rate per year - not unit century}
    n:=n/100;
{Compute the first two Besslian Day numbers using formulae from EPAA}
bea:=n*tau+sin(dtr*EO)*deltaphi;
beb:=-deltaeps;
writeln('Bessel Day number A is:',bea:10:3);
writeln('Bessel Day number B is:',beb:10:3);
{Now find the Bessel Day numbers C and D}
     ts:=T;
if (bis) then epochjdt:=jd+182.625
   else epochjdt:=jd+183.125;
     tb:=(epochjdt-2451545.0)/36525;
     xa:=(2306.2181+1.39656*tb-0.000139*sqr(tb))*ts+(0.30188-0.000344*tb)*sqr(ts)+0.017998*sqr(ts)*ts;
     za:=(2306.2181+1.39656*tb-0.000139*sqr(tb))*ts+(1.09468+0.000066*tb)*sqr(ts)+0.018203*sqr(ts)*ts;
     ta:=(2004.3109-0.85330*tb-0.000217*sqr(tb))*ts+(-0.42665-0.000217*tb)*sqr(ts)-0.041833*sqr(ts)*ts;
{Convert the precession angles from arc seconds -> degrees in prep of computing
precession matrix}
xa:=xa/3600*dtr; za:=za/3600*dtr; ta:=ta/3600*dtr;
{Compute elements of the precession matrix}
         p[1,1]:=cos(za)*cos(ta)*cos(xa)-sin(za)*sin(xa);
         p[1,2]:=-cos(za)*cos(ta)*sin(xa)-sin(za)*cos(xa);
         p[1,3]:=-cos(za)*sin(ta);
         p[2,1]:=sin(za)*cos(ta)*cos(xa)+cos(xa)*sin(xa);
         p[2,2]:=-sin(za)*cos(ta)*sin(xa)+cos(za)*cos(xa);
         p[2,3]:=-sin(za)*sin(ta);
         p[3,1]:=sin(ta)*cos(xa);
         p[3,2]:=-sin(ta)*sin(xa);
         p[3,3]:=cos(ta);

{Convert Barycentric velocities from Au/day -> arc seconds}
kk:=499.004782/86400*rtd*3600;
{Find the velocity components relative to the current date
The lookup table runs from Jan 1 0 UT 1950 -> Jan 1 0 UT 2050
where Jan 1 0 UT 1950 is JD 2433282.5}
tref:=trunc(jd1-2433282.5)+1;
xdotp:=xdot[tref]*p[1,1]+ydot[tref]*p[1,2]+zdot[tref]*p[1,3];
ydotp:=xdot[tref]*p[2,1]+ydot[tref]*p[2,2]+zdot[tref]*p[2,3];
{Calculate C and D Bessel Day Numbers}
c3:=kk*ydotp;
d3:=-kk*xdotp;
writeln('Bessel Day Number C is:',c3:10:5);
writeln('Bessel Day Number D is:',d3:10:5);
writeln('Another (y/n)?'); readln(question);
if (question='y') then finished:=false else finished:=true;
end;
close(ibes);
end.





