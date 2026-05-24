// An-Ra engineering template — stylized turbofan cutaway (NOT OEM-accurate).
// Use as a diagram scaffold; replace dimensions from your source of truth.

$fn = 64;
fan_d = 120;
core_d = 45;
engine_l = 180;

module fan_stage() {
    color("Silver") cylinder(h=12, d=fan_d, center=true);
    for (a = [0:30:330])
        rotate([0, 0, a]) translate([fan_d/2 - 4, 0, 0])
            color("Gray") cube([18, 3, 10], center=true);
}

module core_duct() {
    color("DarkGray") cylinder(h=engine_l - 40, d=core_d, center=true);
}

module exhaust_nozzle() {
    color("Orange") translate([0, 0, engine_l/2 - 20])
        cylinder(h=35, d1=core_d + 10, d2=core_d + 35, center=true);
}

module nacelle() {
    color("LightGray", 0.35) difference() {
        cylinder(h=engine_l, d=fan_d + 25, center=true);
        cylinder(h=engine_l + 2, d=fan_d + 5, center=true);
    }
}

translate([0, 0, 0]) {
    nacelle();
    fan_stage();
    core_duct();
    exhaust_nozzle();
}
