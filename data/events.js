"use strict";
exports.__esModule = true;
var bittrex = require("node-bittrex-api");
bittrex.options({
    'apikey': 'API_KEY',
    'apisecret': 'API_SECRET'
});
// Formats a JSON object into a DBUpdate object
function formatUpdate(v) {
    var updates = [];
    var pair = (v.MarketName);
    var seq = v.Nounce;
    var timestamp = Date.now() / 1000;
    v.Buys.forEach(function (buy) {
        updates.push({
            pair: pair,
            seq: seq,
            is_trade: false,
            is_bid: true,
            price: buy.Rate,
            size: buy.Quantity,
            timestamp: timestamp,
            type: buy.Type
        });
    });
    v.Sells.forEach(function (sell) {
        updates.push({
            pair: pair,
            seq: seq,
            is_trade: false,
            is_bid: false,
            price: sell.Rate,
            size: sell.Quantity,
            timestamp: timestamp,
            type: sell.Type
        });
    });
    v.Fills.forEach(function (fill) {
        updates.push({
            pair: pair,
            seq: seq,
            is_trade: true,
            is_bid: fill.OrderType === "BUY",
            price: fill.Rate,
            size: fill.Quantity,
            timestamp: (new Date(fill.TimeStamp)).getTime() / 1000,
            type: null
        });
    });
    return updates;
}
function watch() {
    try {
        //let mkts = await allMarkets()
        var mkts = ['USDT-BTC'];
        bittrex.websockets.subscribe(mkts, function (data, client) {
            if (data.M === 'updateExchangeState') {
                var state = data;
                state.A.forEach(function (v) {
                    var updates = formatUpdate(v);
                    updates.forEach(function (u) {
                        //console.log(u)
                        console.log(//u.pair + '\t' +
                        u.timestamp + '\t' +
                            u.seq + '\t' +
                            u.size + '\t' +
                            u.price + '\t' +
                            String(+u.is_bid) + '\t' +
                            String(+u.is_trade) + '\t' +
                            u.type);
                    });
                });
            }
        });
    }
    catch (e) {
        console.log(e);
        throw e;
    }
}
var main = watch;
main();
