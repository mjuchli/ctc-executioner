import bittrex = require('node-bittrex-api');
bittrex.options({
  'apikey' : 'API_KEY',
  'apisecret' : 'API_SECRET',
});

export interface ExchangeState {
     H: string, // Hub
     M: "updateExchangeState",
     A: [ExchangeStateUpdate]
}

export type Side = "SELL" | "BUY";
export type UpdateType = 0 // new order entries at matching price, add to orderbook
                       | 1 // cancelled / filled order entries at matching price, delete from orderbook
                       | 2 // changed order entries at matching price (partial fills, cancellations), edit in orderbook
                       ;

export interface ExchangeStateUpdate {
    MarketName: string,
    Nounce: number,
    Buys: [Buy],
    Sells: [Sell],
    Fills: [Fill]
}

export type Sell = Buy;

export interface Buy {
    Type: UpdateType,
    Rate: number,
    Quantity: number
}

export interface Fill {
    OrderType: Side,
    Rate: number,
    Quantity: number,
    TimeStamp: string,
}

//================================

export interface SummaryState {
    H: string,
    M: "updateSummaryState",
    A: [SummaryStateUpdate]
}

export interface SummaryStateUpdate {
    Nounce: number,
    Deltas: [PairUpdate]
}

export interface PairUpdate {
    MarketName: string,
    High: number
    Low: number,
    Volume: number,
    Last: number,
    BaseVolume: number,
    TimeStamp: string,
    Bid: number,
    Ask: number,
    OpenBuyOrders: number,
    OpenSellOrders: number,
    PrevDay: number,
    Created: string
}

//================================

export interface UnhandledData {
    unhandled_data: {
        R: boolean, // true,
        I: string,  // '1'
    }
}

//================================
//callbacks

export type ExchangeCallback = (value: ExchangeStateUpdate, index?: number, array?: ExchangeStateUpdate[]) => void
export type SummaryCallback = (value: PairUpdate, index?: number, array?: PairUpdate[]) => void


//================================
//db updates

export interface DBUpdate {
    pair: string,
    seq: number,
    is_trade: boolean,
    is_bid: boolean,
    price: number,
    size: number,
    timestamp: number,
    type: number
}


// Formats a JSON object into a DBUpdate object
function formatUpdate(v : ExchangeStateUpdate) : DBUpdate[] {
    let updates : DBUpdate[] = [];

    const pair = (v.MarketName);
    const seq = v.Nounce;
    const timestamp = Date.now() / 1000;

    v.Buys.forEach(buy => {
        updates.push(
            {
                pair,
                seq,
                is_trade: false,
                is_bid: true,
                price: buy.Rate,
                size: buy.Quantity,
                timestamp,
                type: buy.Type
            }
        );
    });

    v.Sells.forEach(sell => {
        updates.push(
            {
                pair,
                seq,
                is_trade: false,
                is_bid: false,
                price: sell.Rate,
                size: sell.Quantity,
                timestamp,
                type: sell.Type
            }
        );
    });

    v.Fills.forEach(fill => {
        updates.push(
            {
                pair,
                seq,
                is_trade: true,
                is_bid: fill.OrderType === "BUY",
                price: fill.Rate,
                size: fill.Quantity,
                timestamp: (new Date(fill.TimeStamp)).getTime() / 1000,
                type: null
            }
        );
    })

    return updates;
}

function watch() {
    try {
        //let mkts = await allMarkets()
        let mkts = ['USDT-BTC']
        bittrex.websockets.subscribe(mkts, function(data, client) {
          if (data.M === 'updateExchangeState') {
            const state = <ExchangeState> data;
            state.A.forEach(v => {
              let updates : DBUpdate[] = formatUpdate(v);
              updates.forEach(u => {
                //console.log(u)
                console.log(//u.pair + '\t' +
                            u.timestamp + '\t' +
                            u.seq + '\t' +
                            u.size + '\t' +
                            u.price + '\t' +
                            String(+ u.is_bid) + '\t' +
                            String(+ u.is_trade) + '\t' +
                            u.type
                          )
              });
            })
          }
        });
      } catch (e) {
          console.log(e);
          throw e;
      }
  }

let main = watch;

main();
