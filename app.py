#!/usr/bin/env python
# coding: utf-8

import datetime

import param
import asyncio
import intake
import panel as pn
import numpy as np
import xarray as xr
import xroms
import hvplot.xarray
import geoviews as gv
import geoviews.tile_sources as gts
import cartopy.crs as ccrs
import holoviews as hv
from holoviews.streams import DoubleTap
from bokeh.themes import Theme
from bokeh.models.formatters import PrintfTickFormatter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from asyncio import wrap_future

from dask.distributed import Client
from cluster import DASK_SCHEDULER_ADDRESS



hv.renderer('bokeh').webgl = True
pn.extension(
    throttled=True,
    notifications=True,
    global_loading_spinner=True,
    loading_indicator=True,
    # nthreads=8,
    # defer_load=True,
)
pn.param.ParamMethod.loading_indicator = True

# executor = ThreadPoolExecutor(max_workers=4)
# THEME_JSON = {
#     "attrs": {
#         "figure": {
#             "background_fill_color": "#1b1e23",
#             "border_fill_color": "#1b1e23",
#             "outline_line_alpha": 0,
#         },
#         "Grid": {
#             "grid_line_color": "#808080",
#             "grid_line_alpha": 0.1,
#         },
#         "Axis": {
#             # tick color and alpha
#             "major_tick_line_color": "#4d4f51",
#             "minor_tick_line_alpha": 0,
#             # tick labels
#             "major_label_text_font": "Courier New",
#             "major_label_text_color": "#808080",
#             "major_label_text_align": "left",
#             "major_label_text_font_size": "0.95em",
#             "major_label_text_font_style": "normal",
#             # axis labels
#             "axis_label_text_font": "Courier New",
#             "axis_label_text_font_style": "normal",
#             "axis_label_text_font_size": "1.15em",
#             "axis_label_text_color": "lightgrey",
#             "axis_line_color": "#4d4f51",
#         },
#         "Legend": {
#             "spacing": 8,
#             "glyph_width": 15,
#             "label_standoff": 8,
#             "label_text_color": "#808080",
#             "label_text_font": "Courier New",
#             "label_text_font_size": "0.95em",
#             "label_text_font_style": "bold",
#             "border_line_alpha": 0,
#             "background_fill_alpha": 0.25,
#             "background_fill_color": "#1b1e23",
#         },
#         "BaseColorBar": {
#             # axis labels
#             "title_text_color": "lightgrey",
#             "title_text_font": "Courier New",
#             "title_text_font_size": "0.95em",
#             "title_text_font_style": "normal",
#             # tick labels
#             "major_label_text_color": "#808080",
#             "major_label_text_font": "Courier New",
#             "major_label_text_font_size": "0.95em",
#             "major_label_text_font_style": "normal",
#             "background_fill_color": "#1b1e23",
#             "major_tick_line_alpha": 0,
#             "bar_line_alpha": 0,
#         },
#         "Title": {
#             "text_font": "Courier New",
#             "text_font_style": "normal",
#             "text_color": "lightgrey",
#         },
#     }
# }

# theme = Theme(json=THEME_JSON)
# hv.renderer("bokeh").theme = theme

class COAWST_Viewer(pn.viewable.Viewer):
    time = param.Integer(
        default=0, 
        bounds=(0, 72),
        step=1,
        inclusive_bounds=(True,True),
    )
    depth = param.Selector(
        default="Surface", 
        # label="Network (delete & type to search)"
        objects=["Surface","Middle","Bottom"]
    )
    def __init__(self, **params):
        super().__init__(**params)
        self._sidebar = pn.Column(sizing_mode="stretch_both")
        self._main = pn.Column(
            pn.indicators.LoadingSpinner(
                value=True, min_width=25, min_height=25, name="Loading, Initializing Application..."
            ),
            sizing_mode="stretch_both",
        )
        self._modal = pn.Column(
            min_width=850,
            min_height=500,
            align="center",
            sizing_mode='scale_both'
        )
        self._template = pn.template.FastGridTemplate(
            sidebar=[self._sidebar],
            sidebar_width=250,
            modal=[self._modal],
            # theme="dark",
            theme_toggle=True,
            main_layout=None,
            title="COAWST Viewer",
            # logo='Fathom_logo_horizontal_white.png',
            # site_url='https://www.fathomscience.com/',
            # accent="grey",
            # header_background="#1b1e23",
            row_height=75,
        )
        self._main_title = pn.Column(sizing_mode="stretch_both")
        self._template.main[0,:12] = self._main_title
        self._template.main[1:10,:12] = self._main
        pn.state.onload(self._onload)
        # pn.state.onload(lambda: time_sel.param.trigger("value"))

    def _onload(self):
        try:
            self._sidebar.loading = True
            self._get_client()
            self._populate_sidebar()
            self._populate_modal()
            self._populate_main()
        finally:
            self._sidebar.loading = False
            self._template.open_modal()

    @pn.cache
    def _get_client(self):
        return Client(
            DASK_SCHEDULER_ADDRESS,
            # asynchronous=True
        )
            
    def _populate_sidebar(self):
        open_button = pn.widgets.Button(
            name="About Us",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        open_button.on_click(self._open_modal)
        time_select = pn.widgets.IntSlider.from_param(
            self.param.time, 
            name='Lead',
            format=PrintfTickFormatter(format='+%d hours'),
            sizing_mode='stretch_width',
        )
        depth_select = pn.widgets.Select.from_param(
            self.param.depth,
            name='Depth',
            options=['Surface', 'Middle', 'Bottom'],
            sizing_mode="stretch_width",
        )
        self._salt_pane = pn.pane.HoloViews(
            min_height=100,
        )
        self._wave_pane = pn.pane.HoloViews(
            min_height=100,
        )
        self._current_pane = pn.pane.HoloViews(
            min_height=100,
        )
        self._temperature_pane = pn.pane.HoloViews(
            min_height=100,
        )
        self._sea_level_pane = pn.pane.HoloViews(
            min_height=100,
        )
        self._timeseries = pn.Column(
            self._salt_pane,
            pn.Spacer(max_height=25),
            self._wave_pane,
            pn.Spacer(max_height=25),
            self._current_pane,
            pn.Spacer(max_height=25),
            self._temperature_pane,
            pn.Spacer(max_height=25),
            self._sea_level_pane,
            sizing_mode="scale_both",
        )
        self._sidebar.objects = [
            # pn.pane.Markdown(WELCOME_MESSAGE),
            open_button,
            time_select,
            depth_select,
            self._timeseries,
            # pn.pane.Markdown(FOOTER_MESSAGE),
        ]

    def _populate_main(self):
        pn.state.notifications.info('Application initialized. Loading data...', duration=5000)
        self._main.loading=True
        self._roms_ds = self._get_roms_ds()
        self._time_stamps = self._roms_ds.ocean_time.values.astype(str)
        self._tile = pn.state.as_cached('tile', self._load_tile)
        self._tile_proj = pn.state.as_cached('tile_proj', self._load_tile_proj)
        self._plot_proj = pn.state.as_cached('plot_proj', self._load_plot_proj)
        x,y = self._tile_proj.transform_point(0,0,self._plot_proj)
        self._dtap = DoubleTap(source=self._tile,x=x,y=y)
        pn.state.notifications.info('Data loaded successfully. Creating initial panes...', duration=5000)
        self._markdown_title = pn.pane.Markdown(
            align='center',
            max_height=100,
        )
        self._main_title.objects = [self._markdown_title]
        self._salt_pane.object = pn.bind(self._update_salt_timeseries, self._dtap.param.x, self._dtap.param.y, self.param.time, self.param.depth) 
        self._wave_pane.object = pn.bind(self._update_wave_timeseries, self._dtap.param.x, self._dtap.param.y, self.param.time) 
        self._current_pane.object = pn.bind(self._update_current_timeseries, self._dtap.param.x, self._dtap.param.y, self.param.time, self.param.depth) 
        self._temperature_pane.object = pn.bind(self._update_temperature_timeseries, self._dtap.param.x, self._dtap.param.y, self.param.time, self.param.depth) 
        self._sea_level_pane.object = pn.bind(self._update_sea_level_timeseries, self._dtap.param.x, self._dtap.param.y, self.param.time) 
        self._main.objects = pn.bind(self._update_plots, self.param.time, self.param.depth)

    def _populate_modal(self):
        with open('./README_modal.md') as f:
            modal_txt = f.read()
        modal_txt = pn.pane.Markdown(modal_txt)
        self._modal.objects = [modal_txt]

    def _open_modal(self, event):
        self._template.open_modal()
        
    @pn.cache
    def _get_roms_ds(self):
        intake_catalog_url = 'https://usgs-coawst.s3.amazonaws.com/useast-archive/coawst_intake.yml'
        cat = intake.open_catalog(intake_catalog_url)
        ds = cat['COAWST-USEAST'].to_dask()
        ds=ds[
            ['temp','zeta','u','v','Hwave','Dwave','salt','evaporation']
        ].isel(
            s_rho=[0,7,15],
            ocean_time=slice(-72,None)
        )
        return ds

    def _load_tile(*args, **kwargs):
        return gts.EsriImagery()

    def _load_tile_proj(*args, **kwargs):
        return ccrs.GOOGLE_MERCATOR

    def _load_plot_proj(*args, **kwargs):
        return ccrs.PlateCarree()
    
    @pn.cache(per_session=True)
    def _update_salt_plot(self, time, depth):
        match depth:
            case "Surface":
                depth = -1
            case "Middle":
                depth = -2
            case "Bottom":
                depth = -3
        ds = self._roms_ds.isel(ocean_time=int(time), s_rho=depth)[['salt','evaporation']].persist()
        plot = hv.Overlay([])
        plot *= self._tile
        plot *= ds.salt.hvplot.quadmesh(
            x='lon_rho',y='lat_rho',
            rasterize=True,
            crs=self._plot_proj,
            cmap='pink_r',
            responsive=True,
            clabel='Salinity'
        ).opts(alpha=0.75)
        plot *= ds.evaporation.hvplot.contour(
            x='lon_rho',y='lat_rho',
            crs=self._plot_proj,
            rasterize=True,
            hover=False,
            responsive=True,
        )

        return plot.opts(
            # xlabel=VAR_OPTIONS_R[self.var],
            # ylabel="Number of Days",
            title='Salinity + Contours of Evaporation',
            # shared_axes=False,
            # show_grid=True,
            # gridstyle={"xgrid_line_alpha": 0},
            # xlim=(min_x, df[self.var].max()),
        )

    def _update_salt_timeseries(self,x,y,time,depth):
        roms_spacing=9 #km
        match depth:
            case "Surface":
                depth = -1
            case "Middle":
                depth = -2
            case "Bottom":
                depth = -3
        lon,lat = self._plot_proj.transform_point(x, y,self._tile_proj)
        if ((lon>=-100) and (lon<=-76) and (lat>=18) and (lat<=31)):
            ds = self._roms_ds.isel(s_rho=depth)
            da = ds.where(
                    (ds.lon_rho>=lon-roms_spacing/111) &\
                    (ds.lon_rho<=lon+roms_spacing/111) &\
                    (ds.lat_rho>=lat-roms_spacing/111) &\
                    (ds.lat_rho<=lat+roms_spacing/111)
                ).salt.persist()
            plot = da.mean(dim=['eta_rho','xi_rho'],skipna=True).hvplot(
                    kind='line',
                    x='ocean_time',
                    title=f'Salinity ({lon:.4}, {lat:.4})',
                ).opts(labelled=[],active_tools=[],max_height=200,width=250,yaxis=None)
            plot *= hv.VLine(da.ocean_time.values[int(time)]).opts(color='r')
            return plot

    @pn.cache(per_session=True)
    def _update_wave_plot(self, time):
        ds = self._roms_ds.isel(
            ocean_time=int(time),
            s_rho=-1
        )[["Hwave","Dwave"]].persist()
        plot = hv.Overlay([])
        plot *= self._tile
        plot *= ds.Hwave.hvplot.quadmesh(
            x='lon_rho',y='lat_rho',
            cmap='cool',
            rasterize=True,
            crs=self._plot_proj,
            responsive=True,
            clabel='Height [m]'
        )
        plot *= ds.sel(
            eta_rho=slice(None,None,25),
            xi_rho=slice(None,None,25)
        ).hvplot.vectorfield(
            x='lon_rho',y='lat_rho',
            mag='Hwave',angle='Dwave',
            rasterize=True,hover=False,
            crs=self._plot_proj,
            responsive=True,
        ).opts(magnitude='Hwave',)

        return plot.opts(
            title='Significant Wave Height',
        )

    def _update_wave_timeseries(self,x,y,time):
        roms_spacing=7 #km
        lon,lat = self._plot_proj.transform_point(x, y,self._tile_proj)
        if ((lon>=-100) and (lon<=-76) and (lat>=18) and (lat<=31)):
            ds = self._roms_ds
            da = ds.where(
                    (ds.lon_rho>=lon-roms_spacing/111) &\
                    (ds.lon_rho<=lon+roms_spacing/111) &\
                    (ds.lat_rho>=lat-roms_spacing/111) &\
                    (ds.lat_rho<=lat+roms_spacing/111)
                ).Hwave.persist()
            plot= da.mean(dim=['eta_rho','xi_rho'],skipna=True).hvplot(
                    kind='line',
                    x='ocean_time',
                    title=f'Wave Height m ({lon:.4}, {lat:.4})',
                ).opts(labelled=[],active_tools=[],max_height=200,width=250,yaxis=None)
            plot *= hv.VLine(da.ocean_time.values[int(time)]).opts(color='r')
            return plot

    def _u_to_rho(self,ds,):
        Mp,L=np.shape(ds.u)
        Lp=L+1
        Lm=L-1
        values = 0.5*(ds.u[...,:Lm]+ds.u[...,1:])
        ds["u_rho"]=xr.zeros_like(ds.temp)
        ds["u_rho"][...,1:L]=values.data
        ds["u_rho"][...,0]=ds["u_rho"][...,1]
        ds["u_rho"][...,-1]=ds["u_rho"][...,-2]
        return ds
        
    def _v_to_rho(self,ds):
        M,Lp=np.shape(ds.v)
        Mp=M+1
        Mm=M-1
        values = 0.5*(ds.v[...,:Mm,:]+ds.v[...,1:,:])
        ds["v_rho"]=xr.zeros_like(ds.temp)
        ds["v_rho"][...,1:M,:]=values.data
        ds["v_rho"][...,0,:]=ds["v_rho"][...,1,:]
        ds["v_rho"][...,-1,:]=ds["v_rho"][...,-2,:]
        return ds

    @pn.cache(per_session=True)
    def _update_current_plot(self, time, depth):
        match depth:
            case "Surface":
                depth = -1
            case "Middle":
                depth = -2
            case "Bottom":
                depth = -3
        ds = self._roms_ds.isel(ocean_time=int(time), s_rho=depth)
        ds = self._u_to_rho(ds)
        ds = self._v_to_rho(ds)
        ds["mag"] = np.hypot(ds.u_rho,ds.v_rho)
        ds["angle"] = 3*np.pi/2 - np.arctan2(ds.v_rho, ds.u_rho)
        ds = ds[["mag","angle"]].persist()
        plot = hv.Overlay([])
        plot *= gts.EsriImagery()
        plot *= ds.mag.hvplot.quadmesh(
            x='lon_rho',y='lat_rho',
            rasterize=True,
            crs=self._plot_proj,
            cmap='cet_linear_wcmr_100_45_c42',
            responsive=True,
            clabel='Current Speed [m/s]'
        )
        plot *= ds.sel(
                eta_rho=slice(None,None,25),
                xi_rho=slice(None,None,25)
            ).hvplot.vectorfield(
                x='lon_rho',y='lat_rho',
                mag='mag',angle='angle',
                crs=self._plot_proj,
                rasterize=True,hover=False,
                responsive=True,
        ).opts(magnitude='mag')
        plot *= ds.mag.hvplot.contour(
            x="lon_rho",y="lat_rho",
            levels=[0.75],
            crs=self._plot_proj,
            cmap=['#000000'],
            line_width=2,
            hover=False,
        )

        return plot.opts(
            title='Current Speed',
        )

    def _update_current_timeseries(self,x,y,time,depth):
        roms_spacing=7 #km
        match depth:
            case "Surface":
                depth_idx = -1
            case "Middle":
                depth_idx = -2
            case "Bottom":
                depth_idx = -3
        lon,lat = self._plot_proj.transform_point(x, y,self._tile_proj)
        if ((lon>=-100) and (lon<=-76) and (lat>=18) and (lat<=31)):
            ds = self._roms_ds.isel(s_rho=depth_idx)
            da = ds.where(
                    (ds.lon_rho>=lon-roms_spacing/111) &\
                    (ds.lon_rho<=lon+roms_spacing/111) &\
                    (ds.lat_rho>=lat-roms_spacing/111) &\
                    (ds.lat_rho<=lat+roms_spacing/111) 
                ).mag.persist()
            plot = da.mean(dim=['eta_rho','xi_rho'],skipna=True).hvplot(
                    kind='line',
                    x='ocean_time',
                    title=f'Current Speed m/s ({lon:.4}, {lat:.4})',
                ).opts(labelled=[],active_tools=[],max_height=200,width=250,yaxis=None)
            plot *= hv.VLine(da.ocean_time.values[int(time)]).opts(color='r')
            return plot
    
    @pn.cache(per_session=True)
    def _update_temperature_plot(self, time, depth):
        match depth:
            case "Surface":
                depth = -1
            case "Middle":
                depth = -2
            case "Bottom":
                depth = -3
        ds = self._roms_ds.isel(ocean_time=int(time), s_rho=depth)[["temp","zeta"]].persist()
        plot = hv.Overlay([])
        plot *= gts.EsriImagery()
        plot *= ds.temp.hvplot.quadmesh(
            x='lon_rho',y='lat_rho',
            cmap='Plasma',
            rasterize=True,
            crs=self._plot_proj,
            responsive=True,
            clabel='Temperature [deg C]'
        )
        plot *= ds.zeta.hvplot.contour(
            x='lon_rho',y='lat_rho',
            crs=self._plot_proj,
            cmap=['#000000'],
            line_width=2,
            responsive=True,
        )

        return plot.opts(
            title='Temperature + Sea Level',
        )

    def _update_temperature_timeseries(self,x,y,time,depth):
        roms_spacing=7 #km
        match depth:
            case "Surface":
                depth_idx = -1
            case "Middle":
                depth_idx = -2
            case "Bottom":
                depth_idx = -3
        lon,lat = self._plot_proj.transform_point(x, y,self._tile_proj)
        if ((lon>=-100) and (lon<=-76) and (lat>=18) and (lat<=31)):
            ds = self._roms_ds.isel(s_rho=depth_idx)
            da = ds.where(
                    (ds.lon_rho>=lon-roms_spacing/111) &\
                    (ds.lon_rho<=lon+roms_spacing/111) &\
                    (ds.lat_rho>=lat-roms_spacing/111) &\
                    (ds.lat_rho<=lat+roms_spacing/111)
                ).temp.persist()
            plot = da.mean(dim=['eta_rho','xi_rho'],skipna=True).hvplot(
                    kind='line',
                    x='ocean_time',
                    title=f'Temperature C ({lon:.4}, {lat:.4})',
                ).opts(labelled=[],active_tools=[],max_height=200,width=250,yaxis=None)
            plot *= hv.VLine(da.ocean_time.values[int(time)]).opts(color='r')
            return plot

    def _update_sea_level_timeseries(self,x,y,time):
        roms_spacing=7 #km
        lon,lat = self._plot_proj.transform_point(x, y,self._tile_proj)
        if ((lon>=-100) and (lon<=-76) and (lat>=18) and (lat<=31)):
            ds = self._roms_ds.isel(s_rho=depth_idx)
            da = ds.where(
                    (ds.lon_rho>=lon-roms_spacing/111) &\
                    (ds.lon_rho<=lon+roms_spacing/111) &\
                    (ds.lat_rho>=lat-roms_spacing/111) &\
                    (ds.lat_rho<=lat+roms_spacing/111)
                ).zeta.persist()
            plot = da.mean(dim=['eta_rho','xi_rho'],skipna=True).hvplot(
                    kind='line',
                    x='ocean_time',
                    title=f'Sea Level m ({lon:.4}, {lat:.4})',
                ).opts(labelled=[],active_tools=[],max_height=200,width=250,yaxis=None)
            plot *= hv.VLine(da.ocean_time.values[int(time)]).opts(color='r')
            return plot

    def _update_plots(self,time,depth):
        pn.state.notifications.info('Making plots...', duration=10000)
        try:
            if not self._sidebar.loading:
                self._sidebar.loading = True
            if not self._main.loading:
                self._main.loading = True
            date = self._time_stamps[int(self.time)].split('T')
            self._markdown_title.object = f"## {date[0]} {date[1].split('.')[0]} UTC"
            pn.state.notifications.info('Making salt plot...', duration=30000)
            a=self._update_salt_plot(time,depth)
            pn.state.notifications.info('Making wave plot...', duration=30000)
            b=self._update_wave_plot(time)
            pn.state.notifications.info('Making current plot...', duration=30000)
            c=self._update_current_plot(time,depth)
            pn.state.notifications.info('Making temperature plot...', duration=30000)
            d=self._update_temperature_plot(time,depth)
            pn.state.notifications.info('Rendering...', duration=30000)
            return [pn.Row(a, b, sizing_mode='scale_both'),pn.Row(c,d, sizing_mode='scale_both')]
        finally:
            self._main.loading = False
            self._sidebar.loading = False
    
    def __panel__(self):
        return self._template

COAWST_Viewer().servable()
# COAWST_Viewer()
