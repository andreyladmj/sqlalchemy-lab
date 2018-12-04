

"""
Model
    full update
    partial update
    graph
    get data
    additional html
    additional callbacks
"""
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