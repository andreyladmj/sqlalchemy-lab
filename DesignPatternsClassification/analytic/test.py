

"""
ETL Model
    start date
    timebins period


    full update
    partial update

    graph

    get data
    load partial data from local db
    load full data from local db
    load partial data from remote db
    load full data from remote db
    static data which will not reload but will update

    additional html
    additional callbacks

    resampling
    slice elements on edges
    preprocessing dataframe



    # setup parameters
    metrics_model = OrdersCount

    # resampling parameters
    on = "order_date"
    MIN_BIN_SIZE = 10
    dT = 'W'
    TIMEBIN_PERIOD = pd.Timedelta('7 days 00:00:00')
    date_start = '2015-01-01 00:00:00'
    UPDATE_TIMEBIN_COUNTS = 3  # >= 2

    repository = PaidOrdersCountRepository(period=TIMEBIN_PERIOD,date_start=date_start,time_counts=UPDATE_TIMEBIN_COUNTS)
"""




"""
ETL_GeneralReportGraph
    tag
    label
    additional html
    description
    -
    load from local db
    load from remote db
    get TABLE
    UPDATES
    
    NEED: columns to show on chart, plot params move to model attributes

ETL_LTD_TS_4weeks
    tag
    label
    additional html
    description
    -- RESAMPLING PARAMS --
    on, dT, dT_weeks_count
    -------------
    TIMeBIN PERIOD
    date_start
    TIMEBIN COUNTS
    
    repository = LTDRepository(period=TIMEBIN_PERIOD,date_start=date_start, time_counts=UPDATE_TIMEBIN_COUNTS)

    preprocess dataframe
    resample dataframe
    plot
"""

class ETL_MODEL:
    repository = Repository()












class ETLDirector:
    __builder = None

    def setTimeseriesModelBuilder(self, builder):
        self.__builder = builder

    def getData(self): pass

    def getGraph(self):
        return plot_timeseries(self.__builder.getData())

    def full_update(self): pass
    def partial_update(self): pass


class BuilderInterface:
    def getWheel(self): pass
    def getEngine(self): pass
    def getBody(self): pass

class GeneralReportModelBuilder:
    name_tag = 'tbl_daily_general_report_graph'
    label = "Daily general report Graph"
    filename = name_tag + '.html'
    export_to_file = True
    group_tags = ['all', 'general']
    repository = GeneralReportModelRepository()

    def load_data(self):
        self.data = repository.get_data_from_mysql()

    def getData(self):
        return self.data

    def getAdditionalHtml(self):
        return get_additional_html(self.__class__)
        # return AdditionalHtmlFactory(self)

    def register_additional_callbacks(self):
        pass


d = ETLDirector()
GeneralReportBuilder = GeneralReportModelBuilder()
d.setTimeseriesModelBuilder(GeneralReportBuilder)