Items that cause the level to be skipped in extract_mm2_bcd.py:
    SMB2 Mushroom - Levels with this powerup often fully depend on this and there's no equivalent in SMM: WE
    Track - Levels can easily be made impossible without tracks as they often serve as a way of automation or platforming. They are not in SMM: WE and are just super complicated on their own for models to understand.
    Track Block - Track blocks only work with tracks. We don't want any of them here either.
    Conveyor Belt - Conveyor belts have no implementation in SMM: WE, have weird rules for placements, are a big part of complex levels with automation, and can easily make levels impossible if a jump depends on them. 
    Snake Block - Not in SMM:WE and a snake block can often be the only way to progress in certain levels, so we skip for safety. 

Items that get replaced and with what in json_to_swe.py:
    Fish Bone -> Cheep Cheep
    Seesaw -> Up and Down Platform
    Spike Top -> Spiny
    Slight Slope -> Ground
    Steep Slope -> Ground
    Lemmy, Morton, Larry, Wendy, Iggy, Roy, Ludwig -> Bowser Jr. (still needs to be implemented)

Items that get removed in conversion to .swe:
    Porcupuffer
    Reel Camera
    Sound Effect Marker

All Items Exclusive to 3D World We Skip:
    Stone
    Crate
    Tree
    Blinking Block
    Warp Box
    Clear Pipe
    Piranha Creeper
    Ant Trooper
    Skipsqueak
    Charvaargh
    Bully
    Super Hammer
    Cannon Box
    Propeller Box
    Goomba Mask
    Bullet Bill Mask
    Red Pow Box
    Sprint Platform
    Mushroom Trampoline
    Jumping Machine
    On/Off Trampoline
    Donut Block Platform
    ! Block
    Spike Trap
