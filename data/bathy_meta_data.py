BathyData = {


    ##################################################
    # Kolumbo Data Subset
    ##################################################
    'Kolumbo' : {
        'filepath'  : "/home/gburgess/dvl-nav/data/field_data/kolumbo/bathy/Kolumbo-10m.tif",
        'latlon_format' : True,
        'crop'  : [700, 1501, 700, 1300],
        # 'crop'  : [0, 2000, 0, 2000],
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : 'Kolumbo Volcano, Greece',
        'xlabel': 'Longitude [deg]',
        'ylabel': 'Latitude [deg]',
        'tick_format' : '%.2f',
        'num_ticks' : 3,
        'slope_max' : 50,
        'depth_max' : None,
        'depth_filter' : None,
    },
    
    ##################################################
    # Kolumbo Data Full
    ##################################################
    'Kolumbo_full' : {
        'filepath'  : "/home/gburgess/dvl-nav/data/field_data/kolumbo/bathy/Kolumbo-10m.tif",
        'latlon_format' : True,
        'crop'  : None,
        'name' : 'Kolumbo Volcano, Greece',
        'xlabel': 'Longitude [deg]',
        'ylabel': 'Latitude [deg]',
        'tick_format' : '%.3f',
        'num_ticks' : 3,
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
    },

    ##################################################
    # Puerto Rico Data Full 
    ##################################################
    'Puerto_Rico_full' : {
        'filepath'  : "/home/glider-sim/dvl-nav/data/field_data/puerto_rico/bathy/PuertoRico_SouthWest.tif",
        'latlon_format' : True,
        # 'crop'  : None,
        'crop'  : [700, 5500, 3500, 8500,],
        # 'crop'  : [3500, 4500, 1500, 3500],
        # bathy = bathy_im[top:bot, left:right]
        'name' : 'Puerto Rico',
        'xlabel': 'Longitude [deg]',
        'ylabel': 'Latitude [deg]',
        'tick_format' : '%.2f',
        # 'num_ticks' : None,
        'num_ticks' : 3,
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
    },

    ##################################################
    # Puerto Rico Data Test Site #1
    ##################################################
    'Puerto_Rico_TS1' : {
        'filepath'  : "/home/glider-sim/dvl-nav/data/field_data/puerto_rico/bathy/PuertoRico_SouthWest.tif",
        'latlon_format' : True,
        #'crop'  : [700, 5500, 3500, 8500], FULL
        'crop'  : [900, 2000, 6000, 8500],
        # bathy = bathy_im[top:bot, left:right]
        'name' : 'Puerto Rico',
        'xlabel': 'Longitude [deg]',
        'ylabel': 'Latitude [deg]',
        'tick_format' : '%.2f',
        # 'num_ticks' : None,
        'num_ticks' : 3,
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
    },

    ##################################################
    # Puerto Rico Data Shelf Area
    ##################################################
    'Puerto_Rico_SHELF' : {
        'filepath'  : "/home/glider-sim/dvl-nav/data/field_data/puerto_rico/bathy/PuertoRico_SouthWest.tif",
        'latlon_format' : True,
        #'crop'  : [700, 5500, 3500, 8500], FULL
        #TODO
        'crop'  : [3500, 5000, 2000, 4000],
        # bathy = bathy_im[top:bot, left:right]
        'name' : 'Puerto Rico',
        'xlabel': 'Longitude [deg]',
        'ylabel': 'Latitude [deg]',
        'tick_format' : '%.2f',
        # 'num_ticks' : None,
        'num_ticks' : 3,
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
    },

    ##################################################
    # Puerto Rico Data Off-Shelf Area
    ##################################################
    'Puerto_Rico_OFF_SHELF' : {
        'filepath'  : "/home/glider-sim/dvl-nav/data/field_data/puerto_rico/bathy/PuertoRico_SouthWest.tif",
        'latlon_format' : True,
        #'crop'  : [700, 5500, 3500, 8500], FULL
        'crop'  : [3700, 6500, 2100, 3400],
        # bathy = bathy_im[top:bot, left:right]
        'name' : 'Puerto Rico',
        'xlabel': 'Longitude [deg]',
        'ylabel': 'Latitude [deg]',
        'tick_format' : '%.2f',
        # 'num_ticks' : None,
        'num_ticks' : 3,
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
    },

    ##################################################
    # Buzzards Bay Data
    ##################################################
    'BuzzardsBay' : {
        'filepath'  : "/home/gburgess/dvl-nav/data/field_data/buzz_bay/bathy/BuzzBay_10m.tif",
        'latlon_format'    : False,
        #'crop'  : None,
        'crop'  : [1500, 5740, 1500, 6200],
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : 'Buzzards Bay, MA',
        'xlabel': 'UTM Zone 19',
        'ylabel': '',
        'tick_format' : '%.2g',
        'slope_max' : 8,
        'depth_max' : 35,
        'depth_filter' : None,
        'num_ticks' : 3,
        'meta'  : {
            'utm_zone' : 19,
            'coordinate_system' : 'North American Datum of 1983 and the North American Vertical Datum of 1988',
            'link' : 'https://www.sciencebase.gov/catalog/item/5a4649b8e4b0d05ee8c05486'
        }
    },

    ##################################################
    # Santorini Data Full
    ##################################################
    'Santorini_full' : {
        'filepath'  : "/home/gburgess/dvl-nav/data/field_data/kolumbo/bathy/Christiana-Santorini-Kolumbo.tif",
        'latlon_format'    : True,
        'crop'  : None,
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : 'Kolumbo Volcano, Greece',
        'xlabel': 'Longitude [deg]',
        'ylabel': 'Latitude [deg]',
        'tick_format' : '%.3f',
        'num_ticks' : 3,
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
    },

########################################################################################
# Not kept up to date below
########################################################################################

    ##################################################
    # Costa Rica Data Area1
    ##################################################
    'CostaRica_area1' : {
        'filepath'  : "/Users/zduguid/Dropbox (MIT)/MIT-WHOI/18-Falkor Costa Rica/Bathy for Sentinel survey/Bathy_for_last_Sentinel_missions.tif",
        'latlon_format'    : False,
        'crop'  : None,
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : 'Continental Margin, Costa Rica',
        'xlabel': 'UTM Zone 16',
        'ylabel': '',
        'tick_format' : '%.4g',
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
        'num_ticks' : 3,
        'meta'  : {
            'utm_zone' : '16N',
        }
    },


    ##################################################
    # Costa Rica Data Area3 
    ##################################################
    'CostaRica_area3' : {
        'filepath'  : "/Users/zduguid/Documents/MIT-WHOI/MERS/Cook/cook/bathymetry/jaco-scar-depths.tif",
        'latlon_format'    : False,
        'crop'  : [75, 550, 600, 1200],
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : 'Jaco Scar, Costa Rica',
        'xlabel': 'UTM Zone 16',
        'ylabel': '',
        'tick_format' : '%.4g',
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : 1000,
        'num_ticks' : 3,
        'meta'  : {
            'utm_zone' : '16N',
        }
    },


    ##################################################
    # Costa Rica Data Full
    ##################################################
    'CostaRica_full' : {
        # 'filepath'  : "/Users/zduguid/Documents/MIT-WHOI/MERS/Cook/cook/bathymetry/jaco-scar-depths.tif",
        'filepath'  : "/Users/zduguid/Dropbox (MIT)/MIT-WHOI/18-Falkor Costa Rica/zduguid/three-factor-bathymetry/CostaRica Falkor.tif",
        'latlon_format'    : False,
        'crop'  : False,
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : 'Falkor Dec 2018 Cruise, Costa Rica',
        'xlabel': 'UTM Zone 16',
        'ylabel': '',
        'tick_format' : '%.4g',
        'slope_max' : False,
        'depth_max' : False,
        'depth_filter' : None,
        'num_ticks' : 3,
        'nodata' : 0.0,
        'meta'  : {
            'utm_zone' : '16N',
        }
    },


    ##################################################
    # Hawaii Data Small
    ##################################################
    'Hawaii_small' : {
        'filepath'  : "/Users/zduguid/Documents/MIT-WHOI/MERS/Cook/cook/bathymetry/HI-small.tif",
        'latlon_format'    : True,
        'crop'  : None,
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : "'Au'au Channel, Hawaii",
        'xlabel': 'Lon [deg]',
        'ylabel': 'Lat [deg]',
        'tick_format' : '%.4g',
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
        'num_ticks' : 3,
        'nodata' : None,
        'meta'  : {
            'utm_zone' : '16N',
        }
    },


    ##################################################
    # Hawaii Data Small
    ##################################################
    'Hawaii_all' : {
        'filepath'  : "/Users/zduguid/Documents/MIT-WHOI/MERS/Cook/cook/bathymetry/HI-all.tif",
        'latlon_format'    : True,
        'crop'  : None,
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : "'Au'au Channel, Hawaii",
        'xlabel': 'Lon [deg]',
        'ylabel': 'Lat [deg]',
        'tick_format' : '%.4g',
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,        
        'num_ticks' : 3,
        'nodata' : None,
        'meta'  : {
            'utm_zone' : '16N',
        }
    },


    ##################################################
    # Arctic 400m 
    ##################################################
    'Arctic' : {
        'filepath'  : "/Users/zduguid/Dropbox (MIT)/MIT-WHOI/NSF Arctic NNA/Environment-Data/Arctic-400m/IBCAO_v4_400m.tif",
        'latlon_format'    : False,
        'crop'  : None,
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : 'TODO',
        'xlabel': 'TODO',
        'ylabel': 'TODO',
        'tick_format' : '%.2g',
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
        'num_ticks' : 3,
        'meta'  : None,
    },


    ##################################################
    # Template Data 
    ##################################################
    'template' : {
        'filepath'  : "path/to/file.tif",
        'latlon_format'    : False,
        'crop'  : None,
            # crop  = [top, bot, left, right]
            # bathy = bathy_im[top:bot, left:right]
        'name' : 'TODO',
        'xlabel': 'TODO',
        'ylabel': 'TODO',
        'tick_format' : '%.2g',
        'slope_max' : None,
        'depth_max' : None,
        'depth_filter' : None,
        'num_ticks' : 3,
        'meta'  : None,
    },
}
